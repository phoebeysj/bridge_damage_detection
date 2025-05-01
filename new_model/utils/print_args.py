def safe_str(val):
    return str(val) if val is not None else ""

def print_args(args):
    print("\033[1m" + "Basic Config" + "\033[0m")
    print(f'  {"Task Name:":<20}{safe_str(args.task_name):<20}{"Is Training:":<20}{safe_str(args.is_training):<20}')
    print(f'  {"Model ID:":<20}{safe_str(args.model_id):<20}{"Model:":<20}{safe_str(args.model):<20}')
    print()

    print("\033[1m" + "Data Loader" + "\033[0m")
    print(f'  {"Data:":<20}{safe_str(args.data):<20}{"Root Path:":<20}{safe_str(args.root_path):<20}')
    print(f'  {"Data Path:":<20}{safe_str(args.data_path):<20}{"Features:":<20}{safe_str(args.features):<20}')
    print(f'  {"Target:":<20}{safe_str(args.target):<20}{"Freq:":<20}{safe_str(args.freq):<20}')
    print(f'  {"Checkpoints:":<20}{safe_str(args.checkpoints):<20}')
    print()

    if args.task_name in ['long_term_forecast', 'short_term_forecast']:
        print("\033[1m" + "Forecasting Task" + "\033[0m")
        print(f'  {"Seq Len:":<20}{safe_str(args.seq_len):<20}{"Label Len:":<20}{safe_str(args.label_len):<20}')
        print(f'  {"Pred Len:":<20}{safe_str(args.pred_len):<20}{"Seasonal Patterns:":<20}{safe_str(args.seasonal_patterns):<20}')
        print(f'  {"Inverse:":<20}{safe_str(args.inverse):<20}')
        print()

    if args.task_name == 'imputation':
        print("\033[1m" + "Imputation Task" + "\033[0m")
        print(f'  {"Mask Rate:":<20}{safe_str(args.mask_rate):<20}')
        print()

    if args.task_name == 'anomaly_detection':
        print("\033[1m" + "Anomaly Detection Task" + "\033[0m")
        print(f'  {"Anomaly Ratio:":<20}{safe_str(args.anomaly_ratio):<20}')
        print()

    print("\033[1m" + "Model Parameters" + "\033[0m")
    print(f'  {"Top k:":<20}{safe_str(args.top_k):<20}{"Num Kernels:":<20}{safe_str(args.num_kernels):<20}')
    print(f'  {"Enc In:":<20}{safe_str(args.enc_in):<20}{"Dec In:":<20}{safe_str(args.dec_in):<20}')
    print(f'  {"C Out:":<20}{safe_str(args.c_out):<20}{"d model:":<20}{safe_str(args.d_model):<20}')
    print(f'  {"n heads:":<20}{safe_str(args.n_heads):<20}{"e layers:":<20}{safe_str(args.e_layers):<20}')
    print(f'  {"d layers:":<20}{safe_str(args.d_layers):<20}{"d FF:":<20}{safe_str(args.d_ff):<20}')
    print(f'  {"Moving Avg:":<20}{safe_str(args.moving_avg):<20}{"Factor:":<20}{safe_str(args.factor):<20}')
    print(f'  {"Distil:":<20}{safe_str(args.distil):<20}{"Dropout:":<20}{safe_str(args.dropout):<20}')
    print(f'  {"Embed:":<20}{safe_str(args.embed):<20}{"Activation:":<20}{safe_str(args.activation):<20}')
    print()

    print("\033[1m" + "Run Parameters" + "\033[0m")
    print(f'  {"Num Workers:":<20}{safe_str(args.num_workers):<20}{"Itr:":<20}{safe_str(args.itr):<20}')
    print(f'  {"Train Epochs:":<20}{safe_str(args.train_epochs):<20}{"Batch Size:":<20}{safe_str(args.batch_size):<20}')
    print(f'  {"Patience:":<20}{safe_str(args.patience):<20}{"Learning Rate:":<20}{safe_str(args.learning_rate):<20}')
    print(f'  {"Des:":<20}{safe_str(args.des):<20}{"Loss:":<20}{safe_str(args.loss):<20}')
    print(f'  {"Lradj:":<20}{safe_str(args.lradj):<20}{"Use Amp:":<20}{safe_str(args.use_amp):<20}')
    print()

    print("\033[1m" + "GPU" + "\033[0m")
    print(f'  {"Use GPU:":<20}{safe_str(args.use_gpu):<20}{"GPU:":<20}{safe_str(args.gpu):<20}')
    print(f'  {"Use Multi GPU:":<20}{safe_str(args.use_multi_gpu):<20}{"Devices:":<20}{safe_str(args.devices):<20}')
    print()

    print("\033[1m" + "De-stationary Projector Params" + "\033[0m")
    p_hidden_dims_str = ', '.join(map(str, args.p_hidden_dims)) if hasattr(args, 'p_hidden_dims') and args.p_hidden_dims is not None else ''
    print(f'  {"P Hidden Dims:":<20}{p_hidden_dims_str:<20}{"P Hidden Layers:":<20}{safe_str(args.p_hidden_layers):<20}') 
    print()
