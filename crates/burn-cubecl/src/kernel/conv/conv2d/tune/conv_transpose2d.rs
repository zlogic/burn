use burn_tensor::{ElementConversion, Shape, ops::ConvTransposeOptions};
use cubecl::tune::{LocalTuner, TunableSet, local_tuner};

use crate::{
    CubeAutotuneKey, CubeRuntime, CubeTuneId, FloatElement,
    kernel::{
        conv::{conv_transpose2d_col2im, conv_transpose2d_direct},
        prng::random_uniform,
    },
    tensor::CubeTensor,
};

use super::ConvTranspose2dAutotuneKey;

/// Executes autotune on conv2d operations
pub fn conv_transpose2d_autotune<R: CubeRuntime, E: FloatElement>(
    input: CubeTensor<R>,
    weights: CubeTensor<R>,
    bias: Option<CubeTensor<R>>,
    options: ConvTransposeOptions<2>,
) -> CubeTensor<R> {
    let client = input.client.clone();

    static TUNER: LocalTuner<CubeAutotuneKey, CubeTuneId> = local_tuner!();

    let tune_set = TunableSet::new(create_key::<R, E>, create_transpose2d_input::<R, E>)
        .with_tunable(conv_transpose2d_direct::<R, E>)
        .with_tunable(conv_transpose2d_col2im::<R, E>);

    TUNER
        .execute(
            &CubeTuneId::new::<R>(&input.client, &input.device),
            &client,
            &tune_set,
            (input, weights, bias, options),
        )
        .expect("All autotuners failed")
}

pub fn create_transpose2d_input<R: CubeRuntime, E: FloatElement>(
    key: &CubeAutotuneKey,
    input: &CubeTensor<R>,
    _weights: &CubeTensor<R>,
    _bias: &Option<CubeTensor<R>>,
    options: &ConvTransposeOptions<2>,
) -> (
    CubeTensor<R>,
    CubeTensor<R>,
    Option<CubeTensor<R>>,
    ConvTransposeOptions<2>,
) {
    let key = match key {
        CubeAutotuneKey::ConvTranspose2d(key) => key,
        _ => unreachable!(),
    };
    let device = &input.device;

    let random_bounds: (E, E) = ((-1.0).elem::<E>(), (1.0).elem::<E>());
    let input_shape = Shape::new([key.batch_size, key.in_channels, key.height, key.width]);
    let input = random_uniform(input_shape, device, random_bounds.0, random_bounds.1);
    let c_per_grp = key.in_channels / key.groups;
    let [kernel_h, kernel_w] = key.kernel_size;
    let weight_shape = Shape::new([key.out_channels, c_per_grp, kernel_h, kernel_w]);
    let weights = random_uniform(weight_shape, device, random_bounds.0, random_bounds.1);
    let bias_shape = Shape::new([key.out_channels]);
    let bias = key
        .has_bias
        .then(|| random_uniform(bias_shape, device, random_bounds.0, random_bounds.1));
    (input, weights, bias, options.clone())
}

fn create_key<R: CubeRuntime, E: FloatElement>(
    input: &CubeTensor<R>,
    weights: &CubeTensor<R>,
    bias: &Option<CubeTensor<R>>,
    options: &ConvTransposeOptions<2>,
) -> CubeAutotuneKey {
    let [batch_size, in_channels, height, width] = input.shape.dims();
    let [out_channels, _, kernel_h, kernel_w] = weights.shape.dims();
    let ConvTransposeOptions {
        stride,
        padding,
        dilation,
        groups,
        padding_out,
    } = options.clone();
    CubeAutotuneKey::ConvTranspose2d(ConvTranspose2dAutotuneKey::new(
        [kernel_h, kernel_w],
        stride,
        padding,
        padding_out,
        dilation,
        groups,
        in_channels,
        out_channels,
        height,
        width,
        batch_size,
        bias.is_some(),
        E::dtype(),
    ))
}
