#![allow(missing_docs)]

use burn_tensor::ElementConversion;
use cubecl::{
    AutotuneKey,
    client::ComputeClient,
    tune::{LocalTuner, TunableSet, local_tuner},
};
use serde::{Deserialize, Serialize};

use crate::{
    CubeAutotuneKey, CubeElement, CubeRuntime, CubeTuneId, kernel::prng::random_like_uniform,
    ops::numeric::empty_device, tensor::CubeTensor,
};

/// Executes autotune on reduce operations.
pub fn autotune_reduce<
    Run: CubeRuntime,
    In: CubeElement,
    Out: CubeElement,
    Rd: cubecl::reduce::Reduce,
>(
    client: &ComputeClient<Run::Server, Run::Channel>,
    input: CubeTensor<Run>,
    output: CubeTensor<Run>,
    dim: usize,
) {
    use reduce_ops::*;

    static TUNER: LocalTuner<CubeAutotuneKey, CubeTuneId> = local_tuner!();

    let tunables = TunableSet::new(create_key::<Run>, reduce_input_gen::<Run, In, Out>)
        .with_tunable(reduce::<Run, In, Out, Rd>)
        .with_tunable(reduce_shared::<Run, In, Out, Rd>)
        .with_tunable(reduce_plane::<Run, In, Out, Rd>)
        .with_tunable(reduce_shared_plane::<Run, In, Out, Rd>);

    TUNER
        .execute(
            &CubeTuneId::new::<Run>(&input.client, &input.device),
            client,
            &tunables,
            (input, output, dim),
        )
        .expect("All autotuners failed")
}

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize, AutotuneKey)]
/// Autotune key representative of reduce versions
pub struct ReduceAutotuneKey {
    dtype: burn_tensor::DType,
    #[autotune(anchor)]
    reduce_axis_shape: usize,
    #[autotune(anchor)]
    reduce_axis_stride: usize,
    #[autotune(anchor)]
    outer_axes_product: usize, // The product of the shapes of all axes with greater strides.
}

impl ReduceAutotuneKey {
    pub(crate) fn generate<Run: CubeRuntime>(input: &CubeTensor<Run>, axis: usize) -> Self {
        let rank = input.shape.num_dims();

        if axis > rank {
            panic!("axis {axis} is out-of-bound for a rank of {rank}");
        }

        let dtype = input.dtype;
        let reduce_axis_shape = input.shape.dims[axis];
        let reduce_axis_stride = input.strides[axis];

        let outer_axes_product = input
            .strides
            .iter()
            .zip(input.shape.dims.iter())
            .filter_map(|(stride, shape)| (*stride > reduce_axis_stride).then_some(shape))
            .product();

        Self::new(
            dtype,
            reduce_axis_shape,
            reduce_axis_stride,
            outer_axes_product,
        )
    }
}

pub(crate) fn create_key<Run: CubeRuntime>(
    input: &CubeTensor<Run>,
    _output: &CubeTensor<Run>,
    dim: &usize,
) -> CubeAutotuneKey {
    CubeAutotuneKey::Reduce(ReduceAutotuneKey::generate(input, *dim))
}

mod reduce_ops {
    #![allow(missing_docs)]

    use super::*;

    pub(crate) fn reduce_input_gen<Run: CubeRuntime, In: CubeElement, Out: CubeElement>(
        _key: &CubeAutotuneKey,
        input: &CubeTensor<Run>,
        output: &CubeTensor<Run>,
        dim: &usize,
    ) -> (CubeTensor<Run>, CubeTensor<Run>, usize) {
        let random_bounds: (In, In) = ((-10.0_f32).elem::<In>(), (10.0_f32).elem::<In>());
        let input = random_like_uniform(input, random_bounds.0, random_bounds.1);

        let output = empty_device::<Run, Out>(
            output.client.clone(),
            output.device.clone(),
            output.shape.clone(),
        );

        (input, output, *dim)
    }

    pub(crate) fn reduce<
        Run: CubeRuntime,
        In: CubeElement,
        Out: CubeElement,
        Rd: cubecl::reduce::Reduce,
    >(
        input: CubeTensor<Run>,
        output: CubeTensor<Run>,
        axis: usize,
    ) -> Result<(), String> {
        cubecl::reduce::reduce::<Run, In, Out, Rd>(
            &input.client,
            input.as_handle_ref(),
            output.as_handle_ref(),
            axis,
            Some(cubecl::reduce::ReduceStrategy {
                shared: false,
                use_planes: false,
            }),
        )
        .map_err(|e| format!("{e}"))
    }

    pub(crate) fn reduce_shared<
        Run: CubeRuntime,
        In: CubeElement,
        Out: CubeElement,
        Rd: cubecl::reduce::Reduce,
    >(
        input: CubeTensor<Run>,
        output: CubeTensor<Run>,
        axis: usize,
    ) -> Result<(), String> {
        cubecl::reduce::reduce::<Run, In, Out, Rd>(
            &input.client,
            input.as_handle_ref(),
            output.as_handle_ref(),
            axis,
            Some(cubecl::reduce::ReduceStrategy {
                shared: true,
                use_planes: false,
            }),
        )
        .map_err(|e| format!("{e}"))
    }

    pub(crate) fn reduce_plane<
        Run: CubeRuntime,
        In: CubeElement,
        Out: CubeElement,
        Rd: cubecl::reduce::Reduce,
    >(
        input: CubeTensor<Run>,
        output: CubeTensor<Run>,
        axis: usize,
    ) -> Result<(), String> {
        cubecl::reduce::reduce::<Run, In, Out, Rd>(
            &input.client,
            input.as_handle_ref(),
            output.as_handle_ref(),
            axis,
            Some(cubecl::reduce::ReduceStrategy {
                shared: false,
                use_planes: true,
            }),
        )
        .map_err(|e| format!("{e}"))
    }

    pub(crate) fn reduce_shared_plane<
        Run: CubeRuntime,
        In: CubeElement,
        Out: CubeElement,
        Rd: cubecl::reduce::Reduce,
    >(
        input: CubeTensor<Run>,
        output: CubeTensor<Run>,
        axis: usize,
    ) -> Result<(), String> {
        cubecl::reduce::reduce::<Run, In, Out, Rd>(
            &input.client,
            input.as_handle_ref(),
            output.as_handle_ref(),
            axis,
            Some(cubecl::reduce::ReduceStrategy {
                shared: true,
                use_planes: true,
            }),
        )
        .map_err(|e| format!("{e}"))
    }
}

/// Executes autotune on reduce operations.
#[cfg(feature = "autotune")]
pub fn autotune_sum<Run: CubeRuntime, E: CubeElement>(
    client: &ComputeClient<Run::Server, Run::Channel>,
    input: CubeTensor<Run>,
) -> CubeTensor<Run> {
    use sum_ops::*;

    static TUNER: LocalTuner<CubeAutotuneKey, CubeTuneId> = local_tuner!();

    let tunables = TunableSet::new(create_key_sum::<Run>, sum_input_gen::<Run, E>)
        .with_tunable(sum_chained::<Run, E>)
        .with_tunable(sum_one_shot::<Run, E, 1>)
        .with_tunable(sum_one_shot::<Run, E, 2>)
        .with_tunable(sum_one_shot::<Run, E, 4>)
        .with_tunable(sum_one_shot::<Run, E, 8>)
        .with_tunable(sum_one_shot::<Run, E, 16>)
        .with_tunable(sum_one_shot::<Run, E, 32>)
        .with_tunable(sum_one_shot::<Run, E, 64>);

    TUNER
        .execute(
            &CubeTuneId::new::<Run>(&input.client, &input.device),
            client,
            &tunables,
            input,
        )
        .expect("All autotuners failed")
}

pub(crate) fn create_key_sum<Run: CubeRuntime>(input: &CubeTensor<Run>) -> CubeAutotuneKey {
    CubeAutotuneKey::Sum(SumAutotuneKey::generate(input))
}

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize, AutotuneKey)]
/// Autotune key representative of sum versions
pub struct SumAutotuneKey {
    dtype: burn_tensor::DType,
    #[autotune(anchor)]
    length: usize,
}

impl SumAutotuneKey {
    pub(crate) fn generate<Run: CubeRuntime>(input: &CubeTensor<Run>) -> Self {
        let dtype = input.dtype;
        let length = input.shape.num_elements();
        Self { dtype, length }
    }
}
mod sum_ops {
    #![allow(missing_docs)]

    use cubecl::reduce::instructions::Sum;

    use super::*;

    pub(crate) fn sum_input_gen<Run: CubeRuntime, E: CubeElement>(
        _key: &CubeAutotuneKey,
        input: &CubeTensor<Run>,
    ) -> CubeTensor<Run> {
        let random_bounds: (E, E) = ((-10.0_f32).elem::<E>(), (10.0_f32).elem::<E>());
        random_like_uniform(input, random_bounds.0, random_bounds.1)
    }

    pub(crate) fn sum_one_shot<Run: CubeRuntime, E: CubeElement, const C: u32>(
        input: CubeTensor<Run>,
    ) -> Result<CubeTensor<Run>, String> {
        let client = input.client.clone();
        let device = input.device.clone();
        let handle = client.create(E::as_bytes(&[E::from_int(0)]));
        let output = CubeTensor::new_contiguous(client, device, [1].into(), handle, E::dtype());

        cubecl::reduce::shared_sum::<Run, E>(
            &input.client,
            input.as_handle_ref(),
            output.as_handle_ref(),
            C,
        )
        .map_err(|e| e.to_string())
        .map(|_| output)
    }

    #[cfg(feature = "autotune")]
    pub(crate) fn sum_chained<Run: CubeRuntime, E: CubeElement>(
        input: CubeTensor<Run>,
    ) -> Result<CubeTensor<Run>, String> {
        crate::kernel::reduce::reduce::<Run, E, E, Sum>(
            input,
            crate::kernel::reduce::ReduceStrategy::Autotune,
        )
        .map_err(|e| e.to_string())
    }
}
