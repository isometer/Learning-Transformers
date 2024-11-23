from box import Box
# MSL Train
config={}
config['lr']=1e-4
config['num_epochs']=3
config['k']=3 #heads
config['num_layers']=3 #of encoders and decoders in the transformer
config['win_size']=100
config['input_c']=55
config['output_c']=55 
config['d_ff']=55 # dimension of the feedforward network
config['batch_size']=256
config['dropout']=0.1
config['d_model']=512 # the size of the hidden state vectors
config['pretrained_model']=None
config['dataset']='MSL'
config['mode']='train'
config['data_path']='./dataset/MSL'
config['model_save_path']= 'checkpoints'
config['anormly_ratio']=1
config = Box(config)