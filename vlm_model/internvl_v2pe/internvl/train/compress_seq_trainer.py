from transformers.trainer import *
import sys
from transformers import AutoTokenizer
from transformers.integrations.tpu import tpu_spmd_dataloader

def chunk_with_boundaries(total_length, boundaries, n, chunked_lengths=None):
    boundaries_list = boundaries.tolist()
    for i in range(len(boundaries_list)):
        if boundaries_list[i][-1] != total_length:
            boundaries_list[i].append(total_length)

    chunk_size = total_length // n
    extra = total_length % n

    new_boundaries = [[] for _ in range(n)]

    for batch_idx in range(len(boundaries_list)):
        current_start = 0
        batch_boundaries = boundaries_list[batch_idx]

        for i in range(n):
            if chunked_lengths:
                current_end = current_start + chunked_lengths[i]
            else:
                current_end = current_start + chunk_size + (1 if i < extra else 0)

            chunk_boundary = [max(0, b - current_start) for b in batch_boundaries if current_start <= b <= current_end]
            if len(chunk_boundary)==0:
                chunk_boundary=[0,current_end-current_start]

            if chunk_boundary and chunk_boundary[-1] != current_end - current_start:
                chunk_boundary.append(current_end - current_start)
            # first should be 0
            if chunk_boundary and chunk_boundary[0] != 0:
                chunk_boundary.insert(0, 0)

            new_boundaries[i].append(torch.tensor(chunk_boundary))

            current_start = current_end
    tensor_result=[]
    for i in range(n):
        tensor_result.append(torch.stack(new_boundaries[i], dim=0).to(torch.int32).to(boundaries.device))
    return tensor_result
def extract_local(value, rank, world_size, device, dim=1):
    value_chunks = value.chunk(2 * world_size, dim=dim)
    local_value = torch.cat(
        [value_chunks[rank], value_chunks[2 * world_size - rank - 1]], dim=dim
    )
    return local_value.to(device)


def prepare_zigzag_ring_attn_inputs(
    input_ids, position_ids, target_ids,weights, world_size, device
):
    list1=[]
    list2=[]
    list3=[]
    list4=[]
    for rank in range(world_size):
        local_input_ids = extract_local(
            input_ids,
            rank,
            world_size,
            device,
        )
        local_position_ids = extract_local(
            position_ids,
            rank,
            world_size,
            device,
        )
        if target_ids is not None:
            local_target_ids = extract_local(
                target_ids,
                rank,
                world_size,
                device,
            )
        else:
            local_target_ids = None
        local_weights = extract_local(
            weights,
            rank,
            world_size,
            device
        )
        list1.append(local_input_ids)
        list2.append(local_position_ids)
        list3.append(local_target_ids)
        list4.append(local_weights)
    return list1,list2,list3,list4
def prepare_ulysses_attn_inputs(
    input_ids, position_ids, target_ids,weights, world_size, device
):
    list1=[]
    list2=[]
    list3=[]
    list4=[]
    for rank in range(world_size):
        local_input_ids = extract_local2(
            input_ids,
            rank,
            world_size,
            device,
        )
        local_position_ids = extract_local2(
            position_ids,
            rank,
            world_size,
            device,
        )
        if target_ids is not None:
            local_target_ids = extract_local2(
                target_ids,
                rank,
                world_size,
                device,
            )
        else:
            local_target_ids = None
        local_weights = extract_local2(
            weights,
            rank,
            world_size,
            device
        )
        list1.append(local_input_ids)
        list2.append(local_position_ids)
        list3.append(local_target_ids)
        list4.append(local_weights)
    return list1,list2,list3,list4
def extract_local2(value, rank, world_size, device, dim=1):
    dimension_size = value.shape[dim]
    sub_seq_length = dimension_size // world_size

    sub_seq_start = rank * sub_seq_length
    sub_seq_end = (rank + 1) * sub_seq_length
    local_value = value[:, sub_seq_start:sub_seq_end]

    return local_value.to(device)

def pad_single_inputs(inputs,world_size):
    input_ids = inputs['input_ids']
    loss_weight=torch.tensor(inputs['loss_weight'])
    labels= inputs['labels']
    position_ids=inputs['position_ids']
    if input_ids.shape[1]%(2*world_size)!=0:
        num_padding = 2*world_size-input_ids.shape[1]%(2*world_size)

        padding_shape = (input_ids.shape[0], num_padding)
        input_padding = torch.full(padding_shape, 1, dtype=input_ids.dtype, device=input_ids.device)
        label_padding = torch.full(padding_shape, -100, dtype=labels.dtype, device=inputs['labels'].device)


        input_ids = torch.cat([input_ids, input_padding], dim=1)
        labels = torch.cat([labels, label_padding], dim=1)


        max_pos_id = position_ids.max() + 1
        pos_padding = torch.arange(max_pos_id, max_pos_id + num_padding, device=input_ids.device)
        pos_padding = pos_padding.unsqueeze(0).expand(input_ids.shape[0], -1)
        position_ids = torch.cat([position_ids, pos_padding], dim=1)


        loss_weight_padding = torch.zeros(padding_shape, dtype=loss_weight.dtype, device=loss_weight.device)
        loss_weight = torch.cat([loss_weight, loss_weight_padding], dim=1)
    attn_mask = torch.tensor([[0,input_ids.shape[1]]])
    inputs['input_ids']=input_ids
    inputs['labels']=labels
    inputs['position_ids']=position_ids
    inputs['loss_weight']=list(loss_weight.numpy())
    inputs['attention_mask']=attn_mask
    return inputs
def pad_packed_inputs(inputs,world_size):
    cu_seqlens=inputs['attention_mask']
    assert cu_seqlens.shape[0]==1
    cu_seqlens=cu_seqlens.squeeze(0)
    batch_size = cu_seqlens.shape[0] - 1
    if isinstance(inputs['position_ids'],list):
        is_list=True
        inputs['position_ids']=torch.tensor(inputs['position_ids'])
    # Unpack inputs using cu_seqlens
    unpacked_inputs = []
    for i in range(batch_size):
        start = cu_seqlens[i].item()
        end = cu_seqlens[i + 1].item()
        # Extract individual input 
        single_input = {
            'input_ids': inputs['input_ids'][:, start:end],
            'labels': inputs['labels'][:, start:end],
            'position_ids': inputs['position_ids'][:, start:end],
            'loss_weight': torch.tensor(inputs['loss_weight'])[:, start:end],
        }
        # Pad using pad_single_inputs
        padded_input = pad_single_inputs(single_input, world_size)
        unpacked_inputs.append(padded_input)

    # Re-pack the padded inputs
    packed_input_ids = torch.cat([inp['input_ids'] for inp in unpacked_inputs], dim=1)
    packed_labels = torch.cat([inp['labels'] for inp in unpacked_inputs], dim=1)
    packed_position_ids = torch.cat([inp['position_ids'] for inp in unpacked_inputs], dim=1)
    packed_loss_weight = torch.cat([torch.tensor(inp['loss_weight']) for inp in unpacked_inputs], dim=1)

    # Create new cumulative sequence lengths
    new_cu_seqlens = [0]
    for inp in unpacked_inputs:
        new_cu_seqlens.append(new_cu_seqlens[-1] + inp['input_ids'].shape[1])
    new_cu_seqlens = torch.tensor(new_cu_seqlens, device=inputs['input_ids'].device)
    new_cu_seqlens=new_cu_seqlens.unsqueeze(0).to(torch.int32)
    # Collect keys that don't need padding and include them in the output
    untouched_keys = {k: v for k, v in inputs.items() if k not in 
                      {'input_ids', 'labels', 'position_ids', 'loss_weight', 'attention_mask'}}
    if is_list:
        packed_position_ids=list(packed_position_ids.numpy())
    # Construct the final output
    packed_inputs = {
        'input_ids': packed_input_ids,
        'labels': packed_labels,
        'position_ids': packed_position_ids,
        'loss_weight': list(packed_loss_weight.numpy()),
        'attention_mask': new_cu_seqlens
    }
    # Merge with untouched key-value pairs
    packed_inputs.update(untouched_keys)

    return packed_inputs

class chunkTrainer(Trainer):
    def __init__(self, chunk_num,group_list,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chunk_num = chunk_num
        self.group_list = group_list
        for group_idx,group in enumerate(group_list):
            if type(group)==torch.distributed.distributed_c10d.ProcessGroup:
                print(type(group),'rank',dist.get_rank())
                # assert type(group)==torch.distributed.distributed_c10d.ProcessGroup
                break
        inner_idx=dist.get_rank(group)
        self.group_idx = group_idx
        self.group=group
        self.inner_idx = inner_idx
    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        if self.args.auto_find_batch_size:
            if self.state.train_batch_size != self._train_batch_size:
                from accelerate.utils import release_memory

                (self.model_wrapped,) = release_memory(self.model_wrapped)
                self.model_wrapped = self.model

                # Check for DeepSpeed *after* the intial pass and modify the config
                if self.is_deepspeed_enabled:
                    # Temporarily unset `self.args.train_batch_size`
                    original_bs = self.args.per_device_train_batch_size
                    self.args.per_device_train_batch_size = self._train_batch_size // max(1, self.args.n_gpu)
                    self.propagate_args_to_deepspeed(True)
                    self.args.per_device_train_batch_size = original_bs
            self.state.train_batch_size = self._train_batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()
        if self.is_fsdp_xla_v2_enabled:
            train_dataloader = tpu_spmd_dataloader(train_dataloader)

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        num_train_tokens = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
                if args.include_tokens_per_second:
                    num_train_tokens = (
                        self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
                    )
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
                if args.include_tokens_per_second:
                    num_train_tokens = self.num_tokens(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
            if args.include_tokens_per_second:
                num_train_tokens = self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torchrun or torch.distributed.launch (deprecated))."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState(
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ]
        )
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            if args.gradient_checkpointing_kwargs is None:
                gradient_checkpointing_kwargs = {}
            else:
                gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs

            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

        model = self._wrap_model(self.model_wrapped)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if delay_optimizer_creation:
            if use_accelerator_prepare:
                self._fsdp_qlora_plugin_updates()
                self.model = self.accelerator.prepare(self.model)
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )
        elif self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            # In this case we are in DDP + LOMO, which should be supported
            self.optimizer = self.accelerator.prepare(self.optimizer)

        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # ckpt loading
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(
                    self.model_wrapped, resume_from_checkpoint, load_module_strict=not _is_peft_model(self.model)
                )
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
        # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            self.compare_trainer_and_checkpoint_args(self.args, self.state)
            self._load_callback_state()
            epochs_trained = int(self.state.global_step // num_update_steps_per_epoch)
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()
        grad_norm: Optional[float] = None
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        if args.eval_on_start:
            self._evaluate(trial, ignore_keys_for_eval, skip_scheduler=True)

        total_batched_samples = 0
        for epoch in range(epochs_trained, num_train_epochs):
            epoch_iterator = train_dataloader
            if hasattr(epoch_iterator, "set_epoch"):
                epoch_iterator.set_epoch(epoch)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True


            step = 0
            if not isinstance(epoch_iterator, (list, tuple, dict)):
                epoch_iterator = iter(epoch_iterator)
            while True:
                inputs = None
                
                try:
                    inputs = next(epoch_iterator)
                except StopIteration:
                    inputs = None

                inputs = [inputs]
                dist.broadcast_object_list(inputs, src=self.group_idx*8,group=self.group)
                inputs = inputs[0]

                if inputs is None:
                    break
                if step==0:
                    step+=1
                    continue
                inputs=pad_packed_inputs(inputs,dist.get_world_size(self.group))
                # chunked_inputs=chunk2(inputs, self.chunk_num)
                
                # inputs=chunked_inputs[self.inner_idx]
                
                total_batched_samples += 1

                if self.args.include_num_input_tokens_seen:
                    main_input_name = getattr(self.model, "main_input_name", "input_ids")
                    if main_input_name not in inputs:
                        logger.warning(
                            "Tried to track the number of tokens seen, however the current model is "
                            "not configured properly to know what item is the input. To fix this, add "
                            "a `main_input_name` attribute to the model class you are using."
                        )
                    else:
                        self.state.num_input_tokens_seen += (
                            torch.sum(
                                self.accelerator.gather(
                                    torch.tensor(
                                        inputs[main_input_name].numel(), device=self.args.device, dtype=torch.int64
                                    )
                                )
                            )
                            .cpu()
                            .item()
                        )
                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False


                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    step += 1
                    continue

                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)
                with self.accelerator.accumulate(model):
                    tr_loss_step = self.training_step(model, inputs)
                # if dist.get_rank()==3:
                #     input_text = tokenizer.decode(inputs['input_ids'][0],skip_special_tokens=False)
                #     with open('loss_data_debug.txt','a') as f:
                #         f.write(f"{input_text}\n")
                #         f.write(f"loss:{tr_loss_step.item()},\n")
                #         f.write("$$$$$$$$$\n")
                    
                if (
                    args.logging_nan_inf_filter
                    and not is_torch_xla_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):

                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    if tr_loss.device != tr_loss_step.device:
                        raise ValueError(
                            f"Calculated loss must be on the original device: {tr_loss.device} but device in use is {tr_loss_step.device}"
                        )
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                is_last_step_and_steps_less_than_grad_acc = (
                    steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
                )

                if (
                    total_batched_samples % args.gradient_accumulation_steps == 0
                    or 
                    is_last_step_and_steps_less_than_grad_acc
                ):
                    if is_last_step_and_steps_less_than_grad_acc:
                        self.accelerator.gradient_state._set_sync_gradients(True)

                    if args.max_grad_norm is not None and args.max_grad_norm > 0:
                        if is_sagemaker_mp_enabled() and args.fp16:
                            _grad_norm = self.optimizer.clip_master_grads(args.max_grad_norm)
                        elif self.use_apex:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            _grad_norm = nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer),
                                args.max_grad_norm,
                            )
                        else:
                            _grad_norm = self.accelerator.clip_grad_norm_(
                                model.parameters(),
                                args.max_grad_norm,
                            )

                        if (
                            is_accelerate_available()
                            and self.accelerator.distributed_type == DistributedType.DEEPSPEED
                        ):
                            grad_norm = model.get_global_grad_norm()
                            # In some cases the grad norm may not return a float
                            if hasattr(grad_norm, "item"):
                                grad_norm = grad_norm.item()
                        else:
                            grad_norm = _grad_norm

                    self.optimizer.step()
                    self.control = self.callback_handler.on_optimizer_step(args, self.state, self.control)

                    optimizer_was_run = not self.accelerator.optimizer_step_was_skipped
                    if optimizer_was_run:
                        # Delay optimizer scheduling until metrics are generated
                        if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    # PyTorch/XLA relies on the data loader to insert the mark_step for
                    # each step. Since we are breaking the loop early, we need to manually
                    # insert the mark_step here.
                    if is_torch_xla_available():
                        xm.mark_step()
                    break
                # to_log={}

                # for i in range(len(self.model.language_model.model.layers)):
                #     to_log[f'layer_scale_{i}'] = torch.mean(torch.abs(self.model.language_model.model.layers[i].layer_scale.gamma)).item()
                # to_log[f'rank {dist.get_rank()} loss'] = tr_loss_step.item()
                # print("rank",dist.get_rank(),"loss",tr_loss_step.item())
                # self.log(to_log)
                step += 1
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_xla_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            if is_torch_xla_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        effective_global_step = max(self.state.global_step, 0.001)  # Avoid ZeroDivisionError
        train_loss = self._total_loss_scalar / effective_global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss
        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint, ignore_errors=True)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        return TrainOutput(self.state.global_step, train_loss, metrics)