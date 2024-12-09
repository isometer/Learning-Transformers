from box import Box
# MSL Train
config={}
config['lr']=1e-4
config['num_epochs']=10
config['k']=4 #heads
config['num_layers']=3 #of encoders and decoders in the transformer
config['win_size']=100
config['input_c']=2
config['output_c']=2
config['d_ff']=64 # dimension of the feedforward network; typically 4 * d_model
config['batch_size']=64
config['dropout']=0.2
config['d_model']=16 # the size of the hidden state vectors
config['pretrained_model']=None
config['dataset']='UCI_HAR'
config['mode']='train'
config['data_path']='./data/mit-bih-noise-stress-test-database-1.0.0'
config['model_save_path']= 'checkpoints'
config['anormly_ratio']=1
config = Box(config)