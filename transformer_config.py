from box import Box
# MSL Train
config={}
config['lr']=1e-4
config['num_epochs']=10
config['k']=8 #heads
config['num_layers']=3 #of encoders and decoders in the transformer
config['win_size']=100
config['input_c']=2
config['output_c']=2
config['d_ff']=2048 # dimension of the feedforward network
config['batch_size']=64
config['dropout']=0.2
config['d_model']=512 # the size of the hidden state vectors
config['pretrained_model']=None
config['dataset']='UCI_HAR'
config['mode']='train'
config['data_path']='./data/UCI HAR Dataset'
config['model_save_path']= 'checkpoints'
config['anormly_ratio']=1
config = Box(config)