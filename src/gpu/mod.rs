use std::collections::HashMap;
use ash::{vk, Entry, Device};
use crate::tensor::dequant::QuantTensor;
use crate::gguf::types::GgmlType;

const SPV_Q4_0:    &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/q4_0_gemv.spv"));
const SPV_Q4_1:    &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/q4_1_gemv.spv"));
const SPV_Q4K:     &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/q4k_gemv.spv"));
const SPV_Q3K:     &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/q3k_gemv.spv"));
const SPV_Q5K:     &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/q5k_gemv.spv"));
const SPV_Q6K:     &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/q6k_gemv.spv"));
const SPV_Q8_0:    &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/q8_0_gemv.spv"));
const SPV_F32:     &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/f32_gemv.spv"));
const SPV_RMSNORM: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/rmsnorm.spv"));
const SPV_ROPE:    &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/rope.spv"));
const SPV_KVWRITE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/kv_write.spv"));
const SPV_ATTN:    &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/attention.spv"));
const SPV_SWIGLU:  &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/swiglu.spv"));
const SPV_ADD:     &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/add.spv"));
const SPV_ADD_RN:  &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/add_rmsnorm.spv"));

const SPV_F32_GEMM:    &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/f32_gemm.spv"));
const SPV_Q4_0_GEMM:   &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/q4_0_gemm.spv"));
const SPV_Q4_1_GEMM:   &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/q4_1_gemm.spv"));
const SPV_Q4K_GEMM:    &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/q4k_gemm.spv"));
const SPV_Q3K_GEMM:    &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/q3k_gemm.spv"));
const SPV_Q5K_GEMM:    &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/q5k_gemm.spv"));
const SPV_Q6K_GEMM:    &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/q6k_gemm.spv"));
const SPV_Q8_0_GEMM:   &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/q8_0_gemm.spv"));
const SPV_BATCH_RN:    &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/batch_rmsnorm.spv"));
const SPV_BATCH_ROPE:  &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/batch_rope.spv"));
const SPV_BATCH_KVW:   &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/batch_kv_write.spv"));
const SPV_BATCH_SWGLU: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/batch_swiglu.spv"));
const SPV_BATCH_ADD:   &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/batch_add.spv"));
const SPV_BATCH_BIAS:  &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/batch_bias_add.spv"));
const SPV_BATCH_ATTN:  &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/batch_attn_item.spv"));

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Shader {
    Q4_0, Q4_1, Q4K, Q3K, Q5K, Q6K, Q8_0, F32, RmsNorm, Rope, KvWrite, Attn, SwiGlu, Add, AddRmsNorm,
    // GEMM variants for batched prefill
    GemmF32, GemmQ4_0, GemmQ4_1, GemmQ4K, GemmQ3K, GemmQ5K, GemmQ6K, GemmQ8_0,
    // Batch operation variants
    BatchRmsNorm, BatchRope, BatchKvWrite, BatchSwiGlu, BatchAdd, BatchBiasAdd, BatchAttnItem,
}

pub struct GpuTensor {
    pub buf:    vk::Buffer,
    pub _mem:   vk::DeviceMemory,
    pub rows:   u32,
    pub bpr:    u32,
    pub shader: Shader,
    pub row_start: u32, // offset into backing buffer (for fused QKV slices)
}

pub struct ActBuf {
    pub buf:  vk::Buffer,
    pub mem:  vk::DeviceMemory,
    pub size: u64,
}

pub struct VkCtx {
    _entry:      Entry,
    pub device:  Device,
    queue:       vk::Queue,
    pipes:       HashMap<Shader, (vk::Pipeline, vk::PipelineLayout)>,
    dsl3:        vk::DescriptorSetLayout,
    dsl4:        vk::DescriptorSetLayout,
    dsl5:        vk::DescriptorSetLayout,
    // Per-recording descriptor pool — created in begin_batch/begin, destroyed after submit
    desc_pool:   vk::DescriptorPool,
    cmd_pool:    vk::CommandPool,
    cmd_buf:     vk::CommandBuffer,
    fence:       vk::Fence,
    pub recording: bool,
    pub max_buf: u64,
    pub dev_idx: u32,
    pub host_idx: u32,
    // Persistent staging buffer for per-token embedding uploads
    staging_buf: vk::Buffer,
    _staging_mem: vk::DeviceMemory,
    staging_ptr: *mut u8,
    staging_size: u64,
    ts_pool:     vk::QueryPool,
    ts_period:   f32,
    ts_count:    u32,
    pub debug_gpu: bool,
    // Host-visible persistent staging for batch embeddings (no VRAM used)
    batch_emb_buf:  vk::Buffer,
    batch_emb_mem:  vk::DeviceMemory,
    batch_emb_ptr:  *mut u8,
    batch_emb_size: u64,
    // Keep-alive: prevents GPU from downclocking between token submissions
    ka_cmd:      vk::CommandBuffer,
    ka_fence:    vk::Fence,
    ka_desc_pool: vk::DescriptorPool, // dedicated pool — never destroyed between tokens
    ka_buf:      Option<(vk::Buffer, vk::DeviceMemory)>,
    ka_active:   bool,
}

impl VkCtx {
    pub fn init() -> anyhow::Result<Self> {
        unsafe {
            let entry    = Entry::load()?;
            let app_info = vk::ApplicationInfo {
                api_version: vk::make_api_version(0, 1, 1, 0),
                ..Default::default()
            };
            let instance = entry.create_instance(
                &vk::InstanceCreateInfo::default().application_info(&app_info), None)?;

            let phys_devs = instance.enumerate_physical_devices()?;
            let phys_dev  = phys_devs.iter().copied()
                .max_by_key(|&pd| match instance.get_physical_device_properties(pd).device_type {
                    vk::PhysicalDeviceType::DISCRETE_GPU   => 3,
                    vk::PhysicalDeviceType::INTEGRATED_GPU => 2,
                    vk::PhysicalDeviceType::VIRTUAL_GPU    => 1,
                    _ => 0,
                })
                .ok_or_else(|| anyhow::anyhow!("No Vulkan device"))?;

            let props   = instance.get_physical_device_properties(phys_dev);
            let name    = std::ffi::CStr::from_ptr(props.device_name.as_ptr()).to_string_lossy();
            let max_buf = props.limits.max_storage_buffer_range as u64;
            eprintln!("GPU: {} | max SSBO: {} MB", name, max_buf / 1_000_000);

            let qfams     = instance.get_physical_device_queue_family_properties(phys_dev);
            let queue_fam = qfams.iter().enumerate()
                .find(|(_, f)| f.queue_flags.contains(vk::QueueFlags::COMPUTE))
                .map(|(i, _)| i as u32)
                .ok_or_else(|| anyhow::anyhow!("No compute queue"))?;

            let prio = [1.0f32];
            let qci  = [vk::DeviceQueueCreateInfo::default()
                .queue_family_index(queue_fam).queue_priorities(&prio)];
            let device = instance.create_device(phys_dev,
                &vk::DeviceCreateInfo::default().queue_create_infos(&qci), None)?;
            let queue  = device.get_device_queue(queue_fam, 0);

            let mp       = instance.get_physical_device_memory_properties(phys_dev);
            let dev_idx  = find_mem(&mp, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
            let host_idx = find_mem(&mp,
                vk::MemoryPropertyFlags::HOST_VISIBLE|vk::MemoryPropertyFlags::HOST_COHERENT)?;

            let dsl3 = make_dsl(&device, &[ssbo(0), ssbo(1), ssbo(2)])?;
            let dsl4 = make_dsl(&device, &[ssbo(0), ssbo(1), ssbo(2), ssbo(3)])?;
            let dsl5 = make_dsl(&device, &[ssbo(0), ssbo(1), ssbo(2), ssbo(3), ssbo(4)])?;

            let pc_range = [vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE).offset(0).size(32)];

            let mut pipes = HashMap::new();
            // (shader, spv, which DSL)
            for (shader, spv, dsl) in [
                (Shader::Q4_0,    SPV_Q4_0,    dsl3),
                (Shader::Q4_1,    SPV_Q4_1,    dsl3),
                (Shader::Q4K,     SPV_Q4K,     dsl3),
                (Shader::Q3K,     SPV_Q3K,     dsl3),
                (Shader::Q5K,     SPV_Q5K,     dsl3),
                (Shader::Q6K,     SPV_Q6K,     dsl3),
                (Shader::Q8_0,    SPV_Q8_0,    dsl3),
                (Shader::F32,     SPV_F32,     dsl3),
                (Shader::RmsNorm, SPV_RMSNORM, dsl3),
                (Shader::Rope,    SPV_ROPE,    dsl3),
                (Shader::KvWrite, SPV_KVWRITE, dsl4),
                (Shader::Attn,    SPV_ATTN,    dsl5),
                (Shader::SwiGlu,  SPV_SWIGLU,  dsl3),
                (Shader::Add,     SPV_ADD,     dsl3),
                (Shader::AddRmsNorm, SPV_ADD_RN, dsl4),
                // GEMM shaders (same dsl3 layout as GEMV)
                (Shader::GemmF32,    SPV_F32_GEMM,    dsl3),
                (Shader::GemmQ4_0,   SPV_Q4_0_GEMM,   dsl3),
                (Shader::GemmQ4_1,   SPV_Q4_1_GEMM,   dsl3),
                (Shader::GemmQ4K,    SPV_Q4K_GEMM,    dsl3),
                (Shader::GemmQ3K,    SPV_Q3K_GEMM,    dsl3),
                (Shader::GemmQ5K,    SPV_Q5K_GEMM,    dsl3),
                (Shader::GemmQ6K,    SPV_Q6K_GEMM,    dsl3),
                (Shader::GemmQ8_0,   SPV_Q8_0_GEMM,   dsl3),
                // Batch op shaders
                (Shader::BatchRmsNorm, SPV_BATCH_RN,    dsl3),
                (Shader::BatchRope,    SPV_BATCH_ROPE,  dsl3),
                (Shader::BatchKvWrite, SPV_BATCH_KVW,   dsl4),
                (Shader::BatchSwiGlu,  SPV_BATCH_SWGLU, dsl3),
                (Shader::BatchAdd,     SPV_BATCH_ADD,   dsl3),
                (Shader::BatchBiasAdd, SPV_BATCH_BIAS,  dsl3),
                (Shader::BatchAttnItem,SPV_BATCH_ATTN,  dsl5),
            ] {
                let layout = device.create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .set_layouts(&[dsl]).push_constant_ranges(&pc_range), None)?;
                let module = make_module(&device, spv)?;
                let name   = std::ffi::CString::new("main").unwrap();
                let stage  = vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::COMPUTE).module(module).name(&name);
                let pipe   = device.create_compute_pipelines(vk::PipelineCache::null(),
                    &[vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout)],
                    None).map_err(|(_, e)| e)?[0];
                device.destroy_shader_module(module, None);
                pipes.insert(shader, (pipe, layout));
            }

            let pool_sz   = [vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 10000 }];
            let desc_pool = device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(2000)
                    .pool_sizes(&pool_sz),
                None)?;

            let cmd_pool = device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                    .queue_family_index(queue_fam), None)?;
            let cmd_bufs = device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default().command_pool(cmd_pool)
                    .level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(2))?;
            let cmd_buf = cmd_bufs[0];
            let ka_cmd  = cmd_bufs[1];
            let fence    = device.create_fence(&vk::FenceCreateInfo::default(), None)?;
            let ka_fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;

            // Keep-alive scratch buffer (16KB)
            let ka_size = 16 * 1024u64;
            let ka_buf_obj = device.create_buffer(
                &vk::BufferCreateInfo::default().size(ka_size)
                    .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE), None)?;
            let ka_reqs = device.get_buffer_memory_requirements(ka_buf_obj);
            let ka_mem = device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(ka_reqs.size).memory_type_index(dev_idx), None)?;
            device.bind_buffer_memory(ka_buf_obj, ka_mem, 0)?;

            // Dedicated descriptor pool for keep-alive dispatches (never destroyed)
            let ka_pool_sz = [vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 3 }];
            let ka_desc_pool = device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
                    .max_sets(1)
                    .pool_sizes(&ka_pool_sz),
                None)?;

            // Persistent staging buffer (256KB, persistently mapped) for embedding uploads
            let staging_size = 256 * 1024u64;
            let (staging_buf, _staging_mem) = {
                let buf = device.create_buffer(
                    &vk::BufferCreateInfo::default().size(staging_size)
                        .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE), None)?;
                let reqs = device.get_buffer_memory_requirements(buf);
                let mem = device.allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(reqs.size).memory_type_index(host_idx), None)?;
                device.bind_buffer_memory(buf, mem, 0)?;
                (buf, mem)
            };
            let staging_ptr = device.map_memory(
                _staging_mem, 0, staging_size, vk::MemoryMapFlags::empty())? as *mut u8;

            let ts_period = props.limits.timestamp_period;
            let ts_pool = device.create_query_pool(
                &vk::QueryPoolCreateInfo::default()
                    .query_type(vk::QueryType::TIMESTAMP)
                    .query_count(128), None)?;

            Ok(Self {
                _entry: entry, device, queue, pipes, dsl3, dsl4, dsl5,
                desc_pool, cmd_pool, cmd_buf, fence, recording: false,
                max_buf, dev_idx, host_idx,
                staging_buf, _staging_mem, staging_ptr, staging_size,
                ts_pool, ts_period, ts_count: 0,
                debug_gpu: false,
                batch_emb_buf: vk::Buffer::null(),
                batch_emb_mem: vk::DeviceMemory::null(),
                batch_emb_ptr: std::ptr::null_mut(),
                batch_emb_size: 0,
                ka_cmd, ka_fence, ka_desc_pool, ka_buf: Some((ka_buf_obj, ka_mem)), ka_active: false,
            })
        }
    }

    pub fn upload(&mut self, wt: &QuantTensor) -> Option<GpuTensor> {
        let (packed, shader, bpr) = match wt.typ {
            GgmlType::Q4_0                  => (wt.pack_q4_0_for_gpu(), Shader::Q4_0, (wt.cols/32) as u32),
            GgmlType::Q4_1                  => (wt.pack_q4_1_for_gpu(), Shader::Q4_1, (wt.cols/32) as u32),
            GgmlType::Q4K if wt.cols%256==0 => (wt.pack_q4k_for_gpu(), Shader::Q4K,  (wt.cols/256) as u32),
            GgmlType::Q3K if wt.cols%256==0 => (wt.pack_q3k_for_gpu(), Shader::Q3K,  (wt.cols/256) as u32),
            GgmlType::Q5K if wt.cols%256==0 => (wt.pack_q5k_for_gpu(), Shader::Q5K,  (wt.cols/256) as u32),
            GgmlType::Q6K if wt.cols%256==0 => (wt.pack_q6k_for_gpu(), Shader::Q6K,  (wt.cols/256) as u32),
            GgmlType::Q8_0 if wt.cols%32==0 => (wt.pack_q8_0_for_gpu(),Shader::Q8_0, (wt.cols/32) as u32),
            _ => return None,
        };
        let size = packed.len() as u64 * 4;
        if size > self.max_buf { return None; }
        let (buf, mem) = self.upload_bytes(size,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            bytemuck::cast_slice(&packed)).ok()?;
        Some(GpuTensor { buf, _mem: mem, rows: wt.rows as u32, bpr, shader, row_start: 0 })
    }

    pub fn alloc_act(&mut self, size: u64) -> anyhow::Result<ActBuf> {
        let (buf, mem) = self.alloc_raw(size,
            vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST,
            self.dev_idx)?;
        Ok(ActBuf { buf, mem, size })
    }

    pub fn alloc_readback(&mut self, size: u64) -> anyhow::Result<ActBuf> {
        let (buf, mem) = self.alloc_raw(size, vk::BufferUsageFlags::TRANSFER_DST, self.host_idx)?;
        Ok(ActBuf { buf, mem, size })
    }

    /// Upload CPU float slice into a GPU activation buffer.
    pub fn write_act(&mut self, act: &ActBuf, data: &[f32]) {
        unsafe {
            let size = data.len() as u64 * 4;
            let (stg, sm) = self.alloc_raw(
                size, vk::BufferUsageFlags::TRANSFER_SRC, self.host_idx).unwrap();
            let ptr = self.device.map_memory(sm, 0, size, vk::MemoryMapFlags::empty()).unwrap();
            std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, ptr as *mut u8, size as usize);
            self.device.unmap_memory(sm);
            let cb = self.one_shot_begin();
            self.device.cmd_copy_buffer(cb, stg, act.buf,
                &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size }]);
            self.one_shot_end(cb);
            self.device.destroy_buffer(stg, None);
            self.device.free_memory(sm, None);
        }
    }

    /// Fast embedding upload: memcpy to persistent staging, record copy in main cmd buf.
    /// Must be called after begin() and before any dispatches that read `act`.
    pub fn cmd_upload_act(&mut self, act: &ActBuf, data: &[f32]) {
        unsafe {
            let size = (data.len() * 4) as u64;
            debug_assert!(size <= self.staging_size);
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8, self.staging_ptr, size as usize);
            self.device.cmd_copy_buffer(self.cmd_buf, self.staging_buf, act.buf,
                &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size }]);
            // Barrier: transfer write -> shader read
            let b = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            self.device.cmd_pipeline_barrier(self.cmd_buf,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(), &[b], &[], &[]);
        }
    }

    /// Read logits back to CPU — the copy was already recorded in the main cmd buf,
    /// so we just map the host-visible readback buffer and memcpy.
    pub fn read_logits(&self, _logits_buf: &ActBuf, rb: &ActBuf) -> Vec<f32> {
        unsafe {
            let ptr = self.device.map_memory(rb.mem, 0, rb.size,
                vk::MemoryMapFlags::empty()).unwrap();
            let mut out = vec![0f32; (rb.size / 4) as usize];
            std::ptr::copy_nonoverlapping(
                ptr as *const u8, out.as_mut_ptr() as *mut u8, rb.size as usize);
            self.device.unmap_memory(rb.mem);
            out
        }
    }

    // ── Command encoding ─────────────────────────────────────────────────────

    /// Begin recording for a single token (generation step).
    pub fn begin(&mut self) { self.begin_for_sets(4096); }

    /// Begin recording sized for `n_sets` descriptor sets (prefill batches).
    pub fn begin_for_sets(&mut self, n_sets: u32) {
        unsafe {
            if self.ka_active {
                self.device.wait_for_fences(&[self.ka_fence], true, u64::MAX).unwrap();
                self.device.reset_fences(&[self.ka_fence]).unwrap();
                self.ka_active = false;
            }
            // Destroy previous per-recording pool if any
            if self.desc_pool != vk::DescriptorPool::null() {
                self.device.destroy_descriptor_pool(self.desc_pool, None);
                self.desc_pool = vk::DescriptorPool::null();
            }
            // Create fresh pool sized exactly for this recording
            let pool_sz = [vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: n_sets * 5, // max 5 bindings per DS
            }];
            self.desc_pool = self.device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(n_sets)
                    .pool_sizes(&pool_sz),
                None).unwrap();

            self.device.reset_command_buffer(
                self.cmd_buf, vk::CommandBufferResetFlags::empty()).unwrap();
            self.device.begin_command_buffer(self.cmd_buf,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).unwrap();
            if self.debug_gpu {
                self.device.cmd_reset_query_pool(self.cmd_buf, self.ts_pool, 0, 128);
            }
            self.ts_count = 0;
            self.recording = true;
        }
    }

    pub fn timestamp(&mut self) {
        if !self.debug_gpu { return; }
        if self.ts_count < 128 {
            unsafe {
                self.device.cmd_write_timestamp(
                    self.cmd_buf, vk::PipelineStageFlags::COMPUTE_SHADER,
                    self.ts_pool, self.ts_count);
            }
            self.ts_count += 1;
        }
    }

    pub fn print_timestamps(&self) {
        if !self.debug_gpu || self.ts_count < 2 { return; }
        let mut data = vec![0u64; self.ts_count as usize];
        unsafe {
            self.device.get_query_pool_results(
                self.ts_pool, 0,
                &mut data, vk::QueryResultFlags::TYPE_64).ok();
        }
        let ns = self.ts_period;
        for i in 1..self.ts_count as usize {
            let dt = (data[i].wrapping_sub(data[i-1])) as f64 * ns as f64 / 1_000_000.0;
            eprint!("[ts{}-{}: {:.2}ms] ", i-1, i, dt);
        }
        let total = (data[self.ts_count as usize - 1].wrapping_sub(data[0])) as f64
            * ns as f64 / 1_000_000.0;
        eprintln!("[total GPU: {:.2}ms]", total);
    }

    /// Record the logits->readback copy into the main command buffer, then submit+wait once.
    pub fn submit_with_readback(&mut self, logits_buf: &ActBuf, rb: &ActBuf) {
        unsafe {
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
            self.device.cmd_pipeline_barrier(self.cmd_buf,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(), &[barrier], &[], &[]);
            self.device.cmd_copy_buffer(self.cmd_buf, logits_buf.buf, rb.buf,
                &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: logits_buf.size }]);
            self.device.end_command_buffer(self.cmd_buf).unwrap();
            self.recording = false;
            self.device.queue_submit(self.queue,
                &[vk::SubmitInfo::default().command_buffers(&[self.cmd_buf])],
                self.fence).unwrap();

            // Fire keep-alive immediately after main submit, before wait — GPU stays busy
            // while the CPU is blocked in wait_for_fences + sampling + tokenizer decode.
            // begin_for_sets() waits on ka_fence before the next real submit (correctness safe).
            if let Some((ka_buf, _)) = self.ka_buf {
                self.device.reset_command_buffer(
                    self.ka_cmd, vk::CommandBufferResetFlags::empty()).unwrap();
                self.device.begin_command_buffer(self.ka_cmd,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).unwrap();
                let (pipe, layout) = self.pipes[&Shader::Add];
                // Use dedicated ka_desc_pool — never destroyed, safe across token boundaries
                self.device.reset_descriptor_pool(
                    self.ka_desc_pool, vk::DescriptorPoolResetFlags::empty()).unwrap();
                let ds = alloc_ds(&self.device, self.ka_desc_pool, self.dsl3);
                let i = |b| [vk::DescriptorBufferInfo::default()
                    .buffer(b).offset(0).range(vk::WHOLE_SIZE)];
                let (i0, i1, i2) = (i(ka_buf), i(ka_buf), i(ka_buf));
                self.device.update_descriptor_sets(&[
                    wr(ds, 0, &i0), wr(ds, 1, &i1), wr(ds, 2, &i2),
                ], &[]);
                let pc: [u32; 1] = [4096];
                self.device.cmd_bind_pipeline(
                    self.ka_cmd, vk::PipelineBindPoint::COMPUTE, pipe);
                self.device.cmd_bind_descriptor_sets(
                    self.ka_cmd, vk::PipelineBindPoint::COMPUTE, layout, 0, &[ds], &[]);
                self.device.cmd_push_constants(self.ka_cmd, layout,
                    vk::ShaderStageFlags::COMPUTE, 0,
                    std::slice::from_raw_parts(pc.as_ptr() as *const u8, 4));
                // 128 dispatches covers the ~5-10ms CPU sampling+decode window
                for _ in 0..128 {
                    self.device.cmd_dispatch(self.ka_cmd, 64, 1, 1);
                }
                self.device.end_command_buffer(self.ka_cmd).unwrap();
                self.device.queue_submit(self.queue,
                    &[vk::SubmitInfo::default().command_buffers(&[self.ka_cmd])],
                    self.ka_fence).unwrap();
                self.ka_active = true;
            }

            self.device.wait_for_fences(&[self.fence], true, u64::MAX).unwrap();
            self.device.reset_fences(&[self.fence]).unwrap();
        }
    }


    /// Submit and wait without copying logits — for prefill intermediate chunks.
    pub fn submit_no_readback(&mut self) {
        unsafe {
            self.device.end_command_buffer(self.cmd_buf).unwrap();
            self.recording = false;
            self.device.queue_submit(self.queue,
                &[vk::SubmitInfo::default().command_buffers(&[self.cmd_buf])],
                self.fence).unwrap();
            self.device.wait_for_fences(&[self.fence], true, u64::MAX).unwrap();
            self.device.reset_fences(&[self.fence]).unwrap();
        }
    }

    /// Upload embeddings to a HOST-VISIBLE buffer (system RAM, not VRAM).
    /// This is the correct approach for staging — no VRAM consumed.
    pub fn upload_f32_host_visible(&mut self, data: &[f32]) -> (vk::Buffer, vk::DeviceMemory) {
        unsafe {
            let size = data.len() as u64 * 4;
            let (buf, mem) = self.alloc_raw(
                size,
                vk::BufferUsageFlags::TRANSFER_SRC,
                self.host_idx,
            ).unwrap();
            let ptr = self.device.map_memory(mem, 0, size, vk::MemoryMapFlags::empty()).unwrap();
            std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, ptr as *mut u8, size as usize);
            self.device.unmap_memory(mem);
            (buf, mem)
        }
    }

    /// Upload an f32 weight matrix to a device-local buffer, returns a GpuTensor with F32 shader.
    pub fn upload_f32_weight(&mut self, data: &[f32], rows: u32, cols: u32) -> Option<GpuTensor> {
        let size = data.len() as u64 * 4;
        let (buf, mem) = self.upload_bytes(size,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            bytemuck::cast_slice(data)).ok()?;
        Some(GpuTensor { buf, _mem: mem, rows, bpr: cols, shader: Shader::F32, row_start: 0 })
    }

    /// Record a copy from src+src_offset into act.buf (full act.size bytes) + barrier.
    pub fn cmd_copy_to_act(&self, act: &ActBuf, src: vk::Buffer, src_offset: u64) {
        unsafe {
            self.device.cmd_copy_buffer(self.cmd_buf, src, act.buf,
                &[vk::BufferCopy { src_offset, dst_offset: 0, size: act.size }]);
            let b = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
            self.device.cmd_pipeline_barrier(self.cmd_buf,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(), &[b], &[], &[]);
        }
    }

    /// GEMM: matrix × batch-of-vectors. dispatch = (rows, batch, 1).
    pub fn cmd_gemm(&mut self, wt: &GpuTensor, inp: &ActBuf, out: &ActBuf, batch: u32) {
        let shader = match wt.shader {
            Shader::Q4_0 => Shader::GemmQ4_0,
            Shader::Q4_1 => Shader::GemmQ4_1,
            Shader::Q4K  => Shader::GemmQ4K,
            Shader::Q3K  => Shader::GemmQ3K,
            Shader::Q5K  => Shader::GemmQ5K,
            Shader::Q6K  => Shader::GemmQ6K,
            Shader::Q8_0 => Shader::GemmQ8_0,
            Shader::F32  => Shader::GemmF32,
            _            => return,
        };
        let pc: [u32; 4] = [wt.rows, wt.bpr, batch, wt.row_start];
        let ds = self.ds3(wt.buf, inp.buf, out.buf);
        self.enc3(shader, ds, &pc, wt.rows, batch, 1);
    }

    /// RMSNorm for a batch of vectors. dispatch = (batch, 1, 1).
    pub fn cmd_batch_rmsnorm(&mut self, x: &ActBuf, w: &ActBuf, out: &ActBuf,
                              n: u32, eps: f32, batch: u32) {
        let pc: [u32; 2] = [n, eps.to_bits()];
        let ds = self.ds3(x.buf, w.buf, out.buf);
        self.enc3(Shader::BatchRmsNorm, ds, &pc, batch, 1, 1);
    }

    /// RoPE for a batch of tokens. dispatch proportional to batch × heads × head_dim/2.
    pub fn cmd_batch_rope(&mut self, q: &ActBuf, k: &ActBuf,
                           n_heads: u32, n_kv_heads: u32, head_dim: u32,
                           start_pos: u32, freq: f32, batch: u32) {
        let pc: [u32; 6] = [n_heads, n_kv_heads, head_dim, start_pos, freq.to_bits(), batch];
        let ds = self.ds3(q.buf, k.buf, q.buf);
        let total = batch * (n_heads + n_kv_heads) * (head_dim / 2);
        self.enc3(Shader::BatchRope, ds, bytemuck::cast_slice(&pc), total.div_ceil(64), 1, 1);
    }

    /// Write a batch of K/V vectors to the KV cache.
    pub fn cmd_batch_kv_write(&mut self, k: &ActBuf, v: &ActBuf, kc: &ActBuf, vc: &ActBuf,
                               start_pos: u32, n_kv_heads: u32, head_dim: u32, batch: u32) {
        let kvd = n_kv_heads * head_dim;
        let pc: [u32; 4] = [start_pos, n_kv_heads, head_dim, batch];
        let ds = self.ds4(k.buf, v.buf, kc.buf, vc.buf);
        self.enc4(Shader::BatchKvWrite, ds, bytemuck::cast_slice(&pc), (batch * kvd).div_ceil(64), 1, 1);
    }

    /// SwiGLU over batch × n_ff elements.
    pub fn cmd_batch_swiglu(&mut self, gate: &ActBuf, up: &ActBuf, n_ff: u32, batch: u32) {
        let total = batch * n_ff;
        let pc: [u32; 1] = [total];
        let ds = self.ds3(gate.buf, up.buf, gate.buf);
        self.enc3(Shader::BatchSwiGlu, ds, bytemuck::cast_slice(&pc), total.div_ceil(64), 1, 1);
    }

    /// Element-wise add over batch × n elements (both buffers same size).
    pub fn cmd_batch_add(&mut self, a: &ActBuf, b: &ActBuf, n: u32, batch: u32) {
        let total = batch * n;
        let pc: [u32; 1] = [total];
        let ds = self.ds3(a.buf, b.buf, a.buf);
        self.enc3(Shader::BatchAdd, ds, bytemuck::cast_slice(&pc), total.div_ceil(64), 1, 1);
    }

    /// Broadcast-add a bias vector [n] to each row of a [batch × n] matrix.
    pub fn cmd_batch_bias_add(&mut self, a: &ActBuf, bias: &ActBuf, n: u32, batch: u32) {
        let pc: [u32; 2] = [n, batch];
        let ds = self.ds3(a.buf, bias.buf, a.buf);
        self.enc3(Shader::BatchBiasAdd, ds, bytemuck::cast_slice(&pc), (batch * n).div_ceil(64), 1, 1);
    }

    /// Attention for one token in a batch, reading Q from q_offset and writing to ao_offset.
    pub fn cmd_batch_attn_item(&mut self, q: &ActBuf, kc: &ActBuf, vc: &ActBuf,
                                ao: &ActBuf, scores: &ActBuf,
                                n_heads: u32, n_kv_heads: u32, head_dim: u32,
                                seq_len: u32, n_ctx: u32,
                                q_offset: u32, ao_offset: u32) {
        let pc: [u32; 7] = [n_heads, n_kv_heads, head_dim, seq_len, n_ctx, q_offset, ao_offset];
        let ds = self.ds5(q.buf, kc.buf, vc.buf, ao.buf, scores.buf);
        self.enc5(Shader::BatchAttnItem, ds, bytemuck::cast_slice(&pc), n_heads, 1, 1);
    }

    /// Free a buffer+memory pair after the fence has signalled.
    pub fn free_buffer(&self, buf: vk::Buffer, mem: vk::DeviceMemory) {
        unsafe {
            self.device.destroy_buffer(buf, None);
            self.device.free_memory(mem, None);
        }
    }

    /// Compute-to-compute pipeline barrier.
    pub fn barrier(&self) {
        unsafe {
            let b = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
            self.device.cmd_pipeline_barrier(self.cmd_buf,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(), &[b], &[], &[]);
        }
    }

    pub fn cmd_rmsnorm(&mut self, x: &ActBuf, w: &ActBuf, out: &ActBuf, n: u32, eps: f32) {
        let pc: [u32; 2] = [n, eps.to_bits()];
        let ds = self.ds3(x.buf, w.buf, out.buf);
        self.enc3(Shader::RmsNorm, ds, &pc, 1, 1, 1);
    }

    pub fn cmd_gemv(&mut self, wt: &GpuTensor, inp: &ActBuf, out: &ActBuf) {
        let pc: [u32; 3] = [wt.rows, wt.bpr, wt.row_start];
        let ds = self.ds3(wt.buf, inp.buf, out.buf);
        // 1 workgroup per row — 256 threads cooperate via shared memory reduction
        self.enc3(wt.shader, ds, &pc, wt.rows, 1, 1);
    }

    pub fn cmd_add(&mut self, a: &ActBuf, b: &ActBuf, n: u32) {
        let pc: [u32; 1] = [n];
        let ds = self.ds3(a.buf, b.buf, a.buf);
        self.enc3(Shader::Add, ds, bytemuck::cast_slice(&pc), n.div_ceil(64), 1, 1);
    }

    pub fn cmd_rope(&mut self, q: &ActBuf, k: &ActBuf,
                    n_heads: u32, n_kv_heads: u32, head_dim: u32, pos: u32, freq: f32) {
        let pc: [u32; 5] = [n_heads, n_kv_heads, head_dim, pos, freq.to_bits()];
        let ds = self.ds3(q.buf, k.buf, q.buf);
        let total = (n_heads + n_kv_heads) * (head_dim / 2);
        self.enc3(Shader::Rope, ds, bytemuck::cast_slice(&pc), total.div_ceil(64), 1, 1);
    }

    pub fn cmd_kv_write(&mut self, k: &ActBuf, v: &ActBuf, kc: &ActBuf, vc: &ActBuf,
                         pos: u32, n_kv_heads: u32, head_dim: u32) {
        let kvd = n_kv_heads * head_dim;
        let pc: [u32; 4] = [pos, n_kv_heads, head_dim, 0];
        let ds = self.ds4(k.buf, v.buf, kc.buf, vc.buf);
        self.enc4(Shader::KvWrite, ds, bytemuck::cast_slice(&pc), kvd.div_ceil(64), 1, 1);
    }

    pub fn cmd_attention(&mut self, q: &ActBuf, kc: &ActBuf, vc: &ActBuf,
                          ao: &ActBuf, scores: &ActBuf,
                          n_heads: u32, n_kv_heads: u32, head_dim: u32,
                          seq_len: u32, n_ctx: u32) {
        let pc: [u32; 5] = [n_heads, n_kv_heads, head_dim, seq_len, n_ctx];
        let ds = self.ds5(q.buf, kc.buf, vc.buf, ao.buf, scores.buf);
        self.enc5(Shader::Attn, ds, bytemuck::cast_slice(&pc), n_heads, 1, 1);
    }

    pub fn cmd_swiglu(&mut self, gate: &ActBuf, up: &ActBuf, n: u32) {
        let pc: [u32; 1] = [n];
        let ds = self.ds3(gate.buf, up.buf, gate.buf);
        self.enc3(Shader::SwiGlu, ds, bytemuck::cast_slice(&pc), n.div_ceil(64), 1, 1);
    }

    /// Fused residual add + rmsnorm: res += add_buf, then rmsnorm(res) -> out
    #[allow(dead_code)]
    pub fn cmd_add_rmsnorm(&mut self, res: &ActBuf, add_buf: &ActBuf,
                           w: &ActBuf, out: &ActBuf, n: u32, eps: f32) {
        let pc: [u32; 2] = [n, eps.to_bits()];
        let ds = self.ds4(res.buf, add_buf.buf, w.buf, out.buf);
        self.enc4(Shader::AddRmsNorm, ds, &pc, 1, 1, 1);
    }

    // ── Descriptor set helpers ────────────────────────────────────────────────

    fn ds3(&mut self, b0: vk::Buffer, b1: vk::Buffer, b2: vk::Buffer) -> vk::DescriptorSet {
        unsafe {
            let ds = alloc_ds(&self.device, self.desc_pool, self.dsl3);
            upd3(&self.device, ds, b0, b1, b2);
            ds
        }
    }
    fn ds4(&mut self, b0: vk::Buffer, b1: vk::Buffer,
           b2: vk::Buffer, b3: vk::Buffer) -> vk::DescriptorSet {
        unsafe {
            let ds = alloc_ds(&self.device, self.desc_pool, self.dsl4);
            upd4(&self.device, ds, b0, b1, b2, b3);
            ds
        }
    }
    fn ds5(&mut self, b0: vk::Buffer, b1: vk::Buffer, b2: vk::Buffer,
           b3: vk::Buffer, b4: vk::Buffer) -> vk::DescriptorSet {
        unsafe {
            let ds = alloc_ds(&self.device, self.desc_pool, self.dsl5);
            let bufs = [b0, b1, b2, b3, b4];
            let infos: Vec<[vk::DescriptorBufferInfo; 1]> = bufs.iter()
                .map(|&b| [vk::DescriptorBufferInfo::default().buffer(b).offset(0).range(vk::WHOLE_SIZE)])
                .collect();
            let writes: Vec<vk::WriteDescriptorSet> = (0..5u32)
                .map(|i| vk::WriteDescriptorSet::default()
                    .dst_set(ds).dst_binding(i)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&infos[i as usize]))
                .collect();
            self.device.update_descriptor_sets(&writes, &[]);
            ds
        }
    }

    fn enc3(&self, shader: Shader, ds: vk::DescriptorSet, pc: &[u32], x: u32, y: u32, z: u32) {
        unsafe { self.enc(shader, ds, pc, x, y, z); }
    }
    fn enc4(&self, shader: Shader, ds: vk::DescriptorSet, pc: &[u32], x: u32, y: u32, z: u32) {
        unsafe { self.enc(shader, ds, pc, x, y, z); }
    }
    fn enc5(&self, shader: Shader, ds: vk::DescriptorSet, pc: &[u32], x: u32, y: u32, z: u32) {
        unsafe { self.enc(shader, ds, pc, x, y, z); }
    }
    unsafe fn enc(&self, shader: Shader, ds: vk::DescriptorSet,
                  pc: &[u32], x: u32, y: u32, z: u32) {
        let (pipe, layout) = self.pipes[&shader];
        self.device.cmd_bind_pipeline(self.cmd_buf, vk::PipelineBindPoint::COMPUTE, pipe);
        self.device.cmd_bind_descriptor_sets(
            self.cmd_buf, vk::PipelineBindPoint::COMPUTE, layout, 0, &[ds], &[]);
        self.device.cmd_push_constants(self.cmd_buf, layout,
            vk::ShaderStageFlags::COMPUTE, 0,
            std::slice::from_raw_parts(pc.as_ptr() as *const u8, pc.len() * 4));
        self.device.cmd_dispatch(self.cmd_buf, x, y, z);
    }

    // ── Memory helpers ────────────────────────────────────────────────────────

    unsafe fn one_shot_begin(&self) -> vk::CommandBuffer {
        let cb = self.device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default().command_pool(self.cmd_pool)
                .level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1)).unwrap()[0];
        self.device.begin_command_buffer(cb,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).unwrap();
        cb
    }
    unsafe fn one_shot_end(&self, cb: vk::CommandBuffer) {
        self.device.end_command_buffer(cb).unwrap();
        self.device.queue_submit(self.queue,
            &[vk::SubmitInfo::default().command_buffers(&[cb])], self.fence).unwrap();
        self.device.wait_for_fences(&[self.fence], true, u64::MAX).unwrap();
        self.device.reset_fences(&[self.fence]).unwrap();
        self.device.free_command_buffers(self.cmd_pool, &[cb]);
    }

    fn upload_bytes(&mut self, size: u64, usage: vk::BufferUsageFlags,
                    data: &[u8]) -> anyhow::Result<(vk::Buffer, vk::DeviceMemory)> {
        unsafe {
            let (stg, sm) = self.alloc_raw(
                size, vk::BufferUsageFlags::TRANSFER_SRC, self.host_idx)?;
            let ptr = self.device.map_memory(sm, 0, size, vk::MemoryMapFlags::empty())?;
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut u8, size as usize);
            self.device.unmap_memory(sm);
            let (buf, bm) = self.alloc_raw(
                size, usage | vk::BufferUsageFlags::TRANSFER_DST, self.dev_idx)?;
            let cb = self.one_shot_begin();
            self.device.cmd_copy_buffer(cb, stg, buf,
                &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size }]);
            self.one_shot_end(cb);
            self.device.destroy_buffer(stg, None);
            self.device.free_memory(sm, None);
            Ok((buf, bm))
        }
    }

    /// Grow host-visible embedding staging buffer if needed (no VRAM).
    pub fn ensure_batch_emb_staging(&mut self, needed: u64) {
        if needed <= self.batch_emb_size { return; }
        unsafe {
            if self.batch_emb_size > 0 {
                self.device.unmap_memory(self.batch_emb_mem);
                self.device.destroy_buffer(self.batch_emb_buf, None);
                self.device.free_memory(self.batch_emb_mem, None);
            }
            let (buf, mem) = self.alloc_raw(needed, vk::BufferUsageFlags::TRANSFER_SRC,
                                             self.host_idx).unwrap();
            let ptr = self.device.map_memory(mem, 0, needed, vk::MemoryMapFlags::empty())
                          .unwrap() as *mut u8;
            self.batch_emb_buf  = buf;
            self.batch_emb_mem  = mem;
            self.batch_emb_ptr  = ptr;
            self.batch_emb_size = needed;
        }
    }

    /// CPU memcpy into the batch embedding staging buffer.
    pub fn write_batch_emb(&self, data: &[f32]) {
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8, self.batch_emb_ptr, data.len() * 4);
        }
    }

    /// Returns the host-visible buffer holding staged embeddings.
    pub fn batch_emb_buf(&self) -> vk::Buffer { self.batch_emb_buf }

    pub fn alloc_raw(&self, size: u64, usage: vk::BufferUsageFlags,
                     mt: u32) -> anyhow::Result<(vk::Buffer, vk::DeviceMemory)> {
        unsafe {
            let buf = self.device.create_buffer(
                &vk::BufferCreateInfo::default().size(size).usage(usage)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE), None)?;
            let reqs = self.device.get_buffer_memory_requirements(buf);
            let mem  = self.device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(reqs.size).memory_type_index(mt), None)?;
            self.device.bind_buffer_memory(buf, mem, 0)?;
            Ok((buf, mem))
        }
    }
}

fn find_mem(p: &vk::PhysicalDeviceMemoryProperties,
            f: vk::MemoryPropertyFlags) -> anyhow::Result<u32> {
    (0..p.memory_type_count)
        .find(|&i| p.memory_types[i as usize].property_flags.contains(f))
        .ok_or_else(|| anyhow::anyhow!("No memory type {:?}", f))
}

fn make_dsl(device: &Device,
            bindings: &[vk::DescriptorSetLayoutBinding]) -> anyhow::Result<vk::DescriptorSetLayout> {
    unsafe {
        Ok(device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(bindings), None)?)
    }
}

fn ssbo(b: u32) -> vk::DescriptorSetLayoutBinding<'static> {
    vk::DescriptorSetLayoutBinding::default()
        .binding(b).descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE)
}

unsafe fn alloc_ds(device: &Device, pool: vk::DescriptorPool,
                   layout: vk::DescriptorSetLayout) -> vk::DescriptorSet {
    device.allocate_descriptor_sets(
        &vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool).set_layouts(&[layout])).unwrap()[0]
}

unsafe fn upd3(device: &Device, ds: vk::DescriptorSet,
               b0: vk::Buffer, b1: vk::Buffer, b2: vk::Buffer) {
    let i = |b| [vk::DescriptorBufferInfo::default().buffer(b).offset(0).range(vk::WHOLE_SIZE)];
    let (i0, i1, i2) = (i(b0), i(b1), i(b2));
    device.update_descriptor_sets(&[
        wr(ds, 0, &i0), wr(ds, 1, &i1), wr(ds, 2, &i2),
    ], &[]);
}

unsafe fn upd4(device: &Device, ds: vk::DescriptorSet,
               b0: vk::Buffer, b1: vk::Buffer, b2: vk::Buffer, b3: vk::Buffer) {
    let i = |b| [vk::DescriptorBufferInfo::default().buffer(b).offset(0).range(vk::WHOLE_SIZE)];
    let (i0, i1, i2, i3) = (i(b0), i(b1), i(b2), i(b3));
    device.update_descriptor_sets(&[
        wr(ds, 0, &i0), wr(ds, 1, &i1), wr(ds, 2, &i2), wr(ds, 3, &i3),
    ], &[]);
}

fn wr<'a>(ds: vk::DescriptorSet, b: u32,
          info: &'a [vk::DescriptorBufferInfo]) -> vk::WriteDescriptorSet<'a> {
    vk::WriteDescriptorSet::default()
        .dst_set(ds).dst_binding(b)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .buffer_info(info)
}

unsafe fn make_module(device: &Device, spv: &[u8]) -> anyhow::Result<vk::ShaderModule> {
    let (p, a, s) = spv.align_to::<u32>();
    assert!(p.is_empty() && s.is_empty());
    Ok(device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(a), None)?)
}