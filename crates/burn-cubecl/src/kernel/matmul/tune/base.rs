use burn_tensor::{Element, ElementConversion};
use cubecl::{
    linalg::matmul::{kernels::tiling2d::Tiling2dConfig, Strategy},
    tune::{local_tuner, LocalTuner, TunableSet},
};

use crate::{
    element::FloatElement,
    kernel::{matmul::utils::init_matmul_output, prng::random_like_uniform},
    ops::numeric::empty_device,
    tensor::CubeTensor,
    tune_key::CubeAutotuneKey,
    CubeRuntime, CubeTuneId,
};

use super::key::create_key;

fn matmul_input_gen<R: CubeRuntime, E: FloatElement>(
    _key: &CubeAutotuneKey,
    lhs: &CubeTensor<R>,
    rhs: &CubeTensor<R>,
    out: &CubeTensor<R>,
) -> (CubeTensor<R>, CubeTensor<R>, CubeTensor<R>) {
    let random_bounds: (E, E) = ((-10.0).elem::<E>(), (10.0).elem::<E>());
    let lhs = random_like_uniform(lhs, random_bounds.0, random_bounds.1);
    let rhs = random_like_uniform(rhs, random_bounds.0, random_bounds.1);

    let out = empty_device::<R, E>(out.client.clone(), out.device.clone(), out.shape.clone());

    (lhs, rhs, out)
}

/// Executes autotune on matmul operations
pub fn matmul_autotune<R: CubeRuntime, E: FloatElement + Element>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    out: Option<CubeTensor<R>>,
) -> CubeTensor<R> {
    let output = out.unwrap_or_else(|| init_matmul_output::<R, E>(&lhs, &rhs));

    let client = lhs.client.clone();

    static TUNER: LocalTuner<CubeAutotuneKey, CubeTuneId> = local_tuner!();

    let tunables = TunableSet::new(create_key::<R, E>, matmul_input_gen::<R, E>)
        .with_tunable(matmul_tiling2d::<R, E>)
        .with_tunable(matmul_accelerated::<R, E>)
        .with_tunable(matmul_simple::<R, E>);

    TUNER.execute(
        &CubeTuneId::new::<R>(&lhs.device),
        &client,
        &tunables,
        (lhs, rhs, output.clone()),
    );

    output
}

fn matmul_accelerated<R: CubeRuntime, E: FloatElement>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    out: CubeTensor<R>,
) -> Result<(), String> {
    cubecl::linalg::matmul::launch_ref::<R, E>(
        &Strategy::Simple,
        &lhs.client,
        &lhs.as_handle_ref(),
        &rhs.as_handle_ref(),
        &out.as_handle_ref(),
    )
    .map_err(|err| format!("{err:?}"))
}

fn matmul_tiling2d<R: CubeRuntime, E: FloatElement>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    out: CubeTensor<R>,
) -> Result<(), String> {
    let config = Tiling2dConfig::default();

    let output_shape = out.shape.dims.as_slice();
    let rank = output_shape.len();
    let num_rows = *output_shape.get(rank - 2).unwrap();
    let num_cols = *output_shape.get(rank - 1).unwrap();
    let cubes_x = f32::ceil(num_rows as f32 / config.block_size_m as f32) as u32;
    let cubes_y = f32::ceil(num_cols as f32 / config.block_size_n as f32) as u32;

    let (max_x, max_y, _max_z) = R::max_cube_count();
    if cubes_x > max_x || cubes_y > max_y {
        return Err(format!("Cube size {cubes_x}x{cubes_y} too large"));
    }

    cubecl::linalg::matmul::launch_ref::<R, E>(
        &Strategy::Tiling2D(config),
        &lhs.client,
        &lhs.as_handle_ref(),
        &rhs.as_handle_ref(),
        &out.as_handle_ref(),
    )
    .map_err(|err| format!("{err:?}"))
}

fn matmul_simple<R: CubeRuntime, E: FloatElement>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    out: CubeTensor<R>,
) -> Result<(), String> {
    cubecl::linalg::matmul::launch_ref::<R, E>(
        &Strategy::Simple,
        &lhs.client,
        &lhs.as_handle_ref(),
        &rhs.as_handle_ref(),
        &out.as_handle_ref(),
    )
    .map_err(|err| format!("{err:?}"))
}
