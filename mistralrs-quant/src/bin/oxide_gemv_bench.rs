use candle_core::{cuda::cudarc::driver::sys, Device, Tensor};
use half::bf16;
use mistralrs_quant::gemv::Bf16GemvBenchmarkPlan;
use serde::Serialize;
use serde_json::json;
use std::{env, error::Error};

const CENSUS_SHAPES: [(usize, usize, usize); 5] = [
    (1, 1_536, 1_536),
    (1, 256, 1_536),
    (1, 17_920, 1_536),
    (1, 1_536, 8_960),
    (1, 151_936, 1_536),
];
const FIXTURE_ID: &str = "dyadic_exact_qwen25_15b_gemv_census_v1";
const MEASUREMENT: &str = "eager_stream_batch_cuda_event";
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

#[derive(Clone, Copy)]
struct BenchConfig {
    warmup_launches: usize,
    launches_per_sample: usize,
    samples: usize,
}

impl BenchConfig {
    fn from_env() -> Result<Self, Box<dyn Error>> {
        Ok(Self {
            warmup_launches: env_usize("OXIDE_BENCH_WARMUP", 20)?,
            launches_per_sample: env_usize("OXIDE_BENCH_LAUNCHES", 100)?,
            samples: env_usize("OXIDE_BENCH_SAMPLES", 30)?,
        })
    }
}

struct RunIdentity {
    provider_commit: String,
    run_label: String,
}

impl RunIdentity {
    fn from_env() -> Result<Self, Box<dyn Error>> {
        let provider_commit = env::var("MISTRALRS_SOURCE_COMMIT")?;
        if provider_commit.len() != 40
            || !provider_commit.bytes().all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(
                "MISTRALRS_SOURCE_COMMIT must be a full 40-character Git commit SHA".into(),
            );
        }
        Ok(Self {
            provider_commit,
            run_label: env::var("OXIDE_BENCH_RUN_LABEL")
                .unwrap_or_else(|_| "unlabeled".to_string()),
        })
    }
}

#[derive(Serialize)]
struct BenchmarkRecord<'a> {
    schema_version: u32,
    provider: &'a str,
    provider_version: &'a str,
    provider_commit: &'a str,
    run_label: &'a str,
    measurement: &'a str,
    operator: &'a str,
    case: &'a str,
    dtype: &'a str,
    layout: &'a str,
    execution: serde_json::Value,
    kernels_per_call: usize,
    shape: serde_json::Value,
    fixture_id: &'a str,
    fixture_digests: serde_json::Value,
    warmup_launches: usize,
    launches_per_sample: usize,
    samples_us: Vec<f64>,
}

fn env_usize(name: &str, default: usize) -> Result<usize, Box<dyn Error>> {
    match env::var(name) {
        Ok(value) => {
            let parsed = value.parse::<usize>()?;
            if parsed == 0 {
                Err(format!("{name} must be nonzero").into())
            } else {
                Ok(parsed)
            }
        }
        Err(env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(error.into()),
    }
}

fn exact_activation_value(column: usize) -> f32 {
    ((column % 8) + 1) as f32 / 64.0
}

fn exact_weight_value(row: usize, column: usize) -> f32 {
    (((row * 3 + column * 5) % 16) + 1) as f32 / 256.0
}

fn fixture(n: usize, k: usize) -> (Vec<bf16>, Vec<bf16>) {
    let activation = (0..k)
        .map(|column| bf16::from_f32(exact_activation_value(column)))
        .collect();
    let mut weight = Vec::with_capacity(n * k);
    for row in 0..n {
        weight.extend((0..k).map(|column| bf16::from_f32(exact_weight_value(row, column))));
    }
    (activation, weight)
}

fn reference(activation: &[bf16], weight: &[bf16], n: usize, k: usize) -> Vec<bf16> {
    weight
        .chunks_exact(k)
        .take(n)
        .map(|weight_row| {
            let sum = activation
                .iter()
                .zip(weight_row)
                .fold(0.0_f32, |sum, (&activation, &weight)| {
                    activation.to_f32().mul_add(weight.to_f32(), sum)
                });
            bf16::from_f32(sum)
        })
        .collect()
}

fn digest_bf16(values: &[bf16]) -> u64 {
    values.iter().fold(FNV_OFFSET_BASIS, |digest, value| {
        (digest ^ u64::from(value.to_bits())).wrapping_mul(FNV_PRIME)
    })
}

fn compare_exact(actual: &[bf16], expected: &[bf16]) -> Result<f32, Box<dyn Error>> {
    if actual.len() != expected.len() {
        return Err(format!(
            "matched GEMV output length mismatch: actual={} expected={}",
            actual.len(),
            expected.len()
        )
        .into());
    }
    let mut max_abs = 0.0_f32;
    let mut bit_mismatches = 0_usize;
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let actual_f32 = actual.to_f32();
        if !actual_f32.is_finite() {
            return Err(format!("non-finite matched GEMV output at index {index}").into());
        }
        max_abs = max_abs.max((actual_f32 - expected.to_f32()).abs());
        bit_mismatches += usize::from(actual.to_bits() != expected.to_bits());
    }
    if bit_mismatches != 0 {
        return Err(format!(
            "matched GEMV differed from the exact CPU reference: bit_mismatches={bit_mismatches} max_abs={max_abs}"
        )
        .into());
    }
    Ok(max_abs)
}

fn benchmark_case(
    device: &Device,
    dimensions: (usize, usize, usize),
    config: BenchConfig,
    identity: &RunIdentity,
    device_name: &str,
    compute_capability: (i32, i32),
) -> Result<(), Box<dyn Error>> {
    let (m, n, k) = dimensions;
    if m != 1 {
        return Err("matched Mistral GEMV benchmark requires M=1".into());
    }
    let (activation_host, weight_host) = fixture(n, k);
    let expected = reference(&activation_host, &weight_host, n, k);
    let activation_digest = digest_bf16(&activation_host);
    let weight_digest = digest_bf16(&weight_host);
    let activation = Tensor::from_vec(activation_host, (m, k), device)?;
    let weight = Tensor::from_vec(weight_host, (n, k), device)?;
    let mut plan = Bf16GemvBenchmarkPlan::new(activation, weight)?;

    let Device::Cuda(cuda_device) = device else {
        return Err("matched Mistral GEMV benchmark requires CUDA".into());
    };
    let stream = cuda_device.cuda_stream();
    for _ in 0..config.warmup_launches {
        plan.enqueue()?;
    }
    stream.synchronize()?;

    let start = stream
        .context()
        .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
    let end = stream
        .context()
        .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
    let mut samples_us = Vec::with_capacity(config.samples);
    for _ in 0..config.samples {
        start.record(&stream)?;
        for _ in 0..config.launches_per_sample {
            plan.enqueue()?;
        }
        end.record(&stream)?;
        samples_us
            .push(f64::from(start.elapsed_ms(&end)?) * 1_000.0 / config.launches_per_sample as f64);
    }

    let actual = plan.output()?;
    let max_abs = compare_exact(&actual, &expected)?;
    let output_digest = digest_bf16(&actual);
    let reference_digest = digest_bf16(&expected);
    let case = format!("bf16_gemv_m1_n{n}_k{k}");
    let record = BenchmarkRecord {
        schema_version: 1,
        provider: "mistral.rs",
        provider_version: env!("CARGO_PKG_VERSION"),
        provider_commit: &identity.provider_commit,
        run_label: &identity.run_label,
        measurement: MEASUREMENT,
        operator: "gemv",
        case: &case,
        dtype: "bf16",
        layout: "A_row_major_W_row_major_transposed",
        execution: json!({
            "provider": "Mistral.rs",
            "provider_version": env!("CARGO_PKG_VERSION"),
            "algorithm": "mistral_cuda_gemv_batched",
            "block_size": 256,
            "commands_per_call": 1,
            "workspace_required_bytes": 0,
            "device": device_name,
            "compute_capability": format!("{}.{}", compute_capability.0, compute_capability.1),
            "artifact_target": "sm_90a",
            "correctness": {
                "reference": "CPU F32 sequential dense GEMM",
                "tolerance": "bit_exact_dyadic_fixture",
                "max_abs": max_abs,
                "bit_mismatches": 0,
                "output_digest": format!("{output_digest:016x}"),
                "reference_digest": format!("{reference_digest:016x}"),
            }
        }),
        kernels_per_call: 1,
        shape: json!({"m": m, "n": n, "k": k}),
        fixture_id: FIXTURE_ID,
        fixture_digests: json!({
            "activation": format!("{activation_digest:016x}"),
            "weight_storage": format!("{weight_digest:016x}"),
        }),
        warmup_launches: config.warmup_launches,
        launches_per_sample: config.launches_per_sample,
        samples_us,
    };
    println!("{}", serde_json::to_string(&record)?);
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let config = BenchConfig::from_env()?;
    let identity = RunIdentity::from_env()?;
    if env::var("CUDA_COMPUTE_CAP")?.as_str() != "90" {
        return Err("oxide_gemv_bench requires CUDA_COMPUTE_CAP=90".into());
    }
    let device = Device::new_cuda(0)?;
    let Device::Cuda(cuda_device) = &device else {
        return Err("oxide_gemv_bench requires CUDA device zero".into());
    };
    let context = cuda_device.cuda_stream().context().clone();
    let device_name = context.name()?;
    let compute_capability = context.compute_capability()?;
    if device_name != "NVIDIA H20" || compute_capability != (9, 0) {
        return Err(format!(
            "oxide_gemv_bench admits NVIDIA H20 compute capability 9.0 only, got {device_name} {}.{}",
            compute_capability.0, compute_capability.1
        )
        .into());
    }
    for dimensions in CENSUS_SHAPES {
        benchmark_case(
            &device,
            dimensions,
            config,
            &identity,
            &device_name,
            compute_capability,
        )?;
    }
    Ok(())
}
