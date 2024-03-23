use half::f16;
use rayon::prelude::*;

use crate::backends::cpu::buf::buf_f16::quantize_f32_f16;
use crate::backends::cpu::buf::buf_f16::vec_dot_f16_f16;
use crate::backends::cpu::buf::buf_f16::vec_fma_f16_f16;
use crate::backends::cpu::buf::buf_f32::vec_dot_f32_f32_strided;
use crate::backends::cpu::buf::CpuTensorBuf;
use crate::backends::cpu::CpuTensorDeviceRef;
use crate::tensor::TensorStrider;

// (m, k) @ (k, ) -> (m, ) => gemv_2d_1d
// (s, m, k) @ (s, k, ) -> (s, m, )  => bmm_3d_2d
// (b, s, m, k) @ (b, s, k) -> (b, s, m) => bmm_4d_3d
// a is allowed to be not contiguous, but not quantized
pub fn gemv<'a>(
    device: &CpuTensorDeviceRef<'a>,
    a: &CpuTensorBuf<'a>,
    b: &CpuTensorBuf<'a>,
    c: &mut CpuTensorBuf<'a>,
    strider1: &TensorStrider,
    strider2: &TensorStrider,
) {
    assert!(strider1.dims() == 3 || strider1.dims() == 2);
    assert!(strider2.dims() == strider1.dims() - 1);
    assert!(strider2.is_contiguous());

    match strider1.dims() {
        2 => {
            assert!(strider1.is_contiguous());
            assert!(strider1.shape()[1] == strider2.shape()[0]);

            let (m, k) = (strider1.shape()[0], strider1.shape()[1]);
            let (mi_stride, ki_stride) = (strider1.strides()[0], strider1.strides()[1]);
            gemv_dense_3d_2d(device, a, b, c, 1, 1, m, k, 0, mi_stride, ki_stride);
        }

        3 => {
            assert!(strider1.shape()[2] == strider2.shape()[1]);

            let (a_batch, b_batch) = (strider1.shape()[0], strider2.shape()[0]);
            let (m, k) = (strider1.shape()[1], strider1.shape()[2]);
            let (bi_stride, mi_stride, ki_stride) = (
                strider1.strides()[0],
                strider1.strides()[1],
                strider1.strides()[2],
            );
            match a {
                CpuTensorBuf::F32(bufa) => {
                    let bufc = c.as_f32_mut();
                    let bufb = b.as_f32_ref();
                    gemv_3d_2d_f32(
                        device, bufa, bufb, bufc, a_batch, b_batch, m, k, bi_stride, mi_stride,
                        ki_stride,
                    );
                }
                CpuTensorBuf::F16(bufa) => {
                    let bufb = b.as_f32_ref();
                    let bufc = c.as_f32_mut();
                    let bufb = quantize_f32_f16(bufb);
                    gemv_3d_2d_f16(
                        device, bufa, &bufb, bufc, a_batch, b_batch, m, k, bi_stride, mi_stride,
                        ki_stride,
                    );
                }
                bufa => {
                    assert!(strider1.is_contiguous());
                    gemv_dense_3d_2d(
                        device, bufa, b, c, a_batch, b_batch, m, k, bi_stride, mi_stride, ki_stride,
                    );
                }
            }
        }
        _ => unreachable!(),
    }
}

fn gemv_dense_3d_2d(
    _device: &CpuTensorDeviceRef,
    bufa: &CpuTensorBuf,
    bufb: &CpuTensorBuf,
    bufc: &mut CpuTensorBuf,
    a_batch: usize,
    _b_batch: usize, // b_batch is multiple of a_batch
    m: usize,
    k: usize,
    bi_stride: usize,
    mi_stride: usize,
    _ki_stride: usize,
) {
    let bufc = bufc.as_f32_mut();
    let bufb = &bufb.quantize(bufa.vec_dot_rhs_dtype()).unwrap();
    bufc.par_chunks_exact_mut(4)
        .enumerate()
        .for_each(|(cn, cp)| {
            // a: b x m x k
            // b: b x k
            // c: b x m
            let mi = cn * 4 % m;
            let bi = (cn * 4 - mi) / m;
            let bi_offset = (bi % a_batch) * bi_stride;
            cp[0] = bufa.vec_dot(bi_offset + mi * mi_stride, bufb, 0, k);
            cp[1] = bufa.vec_dot(bi_offset + (mi + 1) * mi_stride, bufb, 0, k);
            cp[2] = bufa.vec_dot(bi_offset + (mi + 2) * mi_stride, bufb, 0, k);
            cp[3] = bufa.vec_dot(bi_offset + (mi + 3) * mi_stride, bufb, 0, k);
        });
}

#[allow(clippy::too_many_arguments)]
fn gemv_3d_2d_f32(
    _device: &CpuTensorDeviceRef,
    abuf: &[f32],     // a_batch x M x K
    bbuf: &[f32],     // b_batch x K
    cbuf: &mut [f32], // b_batch x M
    a_batch: usize,
    _b_batch: usize,
    m: usize,
    k: usize,
    bi_stride: usize,
    mi_stride: usize,
    ki_stride: usize,
) {
    cbuf.par_iter_mut().enumerate().for_each(|(i, bufcp)| {
        let mi = i % m;
        let bi = (i - mi) / m;
        *bufcp = vec_dot_f32_f32_strided(
            abuf,
            (bi % a_batch) * bi_stride + mi * mi_stride,
            ki_stride,
            k,
            &bbuf[bi * k..(bi + 1) * k],
        );
    });
}

#[allow(clippy::too_many_arguments)]
fn gemv_3d_2d_f16(
    device: &CpuTensorDeviceRef,
    abuf: &[f16],     // Batch x M x K
    bbuf: &[f16],     // Batch x K
    cbuf: &mut [f32], // Batch x M
    a_batch: usize,
    b_batch: usize, // b_batch is multiple of a_batch
    m: usize,
    k: usize,
    a_stride0: usize,
    a_stride1: usize,
    a_stride2: usize,
) {
    let mut tmpc = vec![f16::ZERO; b_batch * m]; // TODO: avoid allocation

    if a_stride2 == 1 {
        let _t = device.metrics.batch_matmul_rowwise_walltime.track();
        tmpc.par_iter_mut().enumerate().for_each(|(i, bufcp)| {
            let mi = i % m;
            let bi = (i - mi) / m;
            *bufcp = f16::from_f32(vec_dot_f16_f16(
                abuf,
                (bi % a_batch) * a_stride0 + mi * a_stride1,
                &bbuf[bi * k..(bi + 1) * k],
                0,
                k,
            ));
        });
    } else {
        let _t = device.metrics.batch_matmul_colwise_walltime.track();
        for bi in 0..b_batch {
            for ki in 0..k {
                vec_fma_f16_f16(
                    abuf,
                    bbuf[bi * k + ki],
                    &mut tmpc[bi * m..],
                    (bi % a_batch) * a_stride0 + ki * a_stride2,
                    m,
                );
            }
        }
    }

    cbuf.iter_mut().zip(tmpc.iter()).for_each(|(c, tmp)| {
        *c = tmp.to_f32();
    });
}
