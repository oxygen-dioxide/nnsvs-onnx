import os
import sys
sys.path.append(os.path.dirname(__file__))

import argparse
from datetime import datetime
import enulib
from enulib.common import set_checkpoint, set_normalization_stat
import hydra
from hydra.utils import to_absolute_path
import joblib
import json
from nnmnkwii.io import hts
import numpy
from omegaconf import DictConfig, OmegaConf
import sklearn
import torch
from typing import List,Dict
import nnsvs
from nnsvs.base import PredictionType

modeltype = None

def to_one_hot(tensor, n, fill_with=1.0, lengths=None):
    # we perform one hot encore with respect to the last axis
    if(lengths == None):
        one_hot = torch.FloatTensor(tensor.size() + torch.Size((n,))).zero_()
    elif(modeltype == nnsvs.model.RMDN):
        one_hot = torch.zeros(
            (
                1,
                lengths.squeeze(),
                n.item()
            ),
            dtype=torch.float)
    elif(modeltype == nnsvs.model.MDNv2):
        one_hot = torch.zeros(
            (
                1,
                lengths.squeeze(),
                1,
                n.item()
            ),
            dtype=torch.float)
    else:
        raise Exception("Model type not supported")
    if tensor.is_cuda:
        one_hot = one_hot.cuda()
    one_hot.scatter_(len(tensor.size()), tensor.unsqueeze(-1), fill_with)
    return one_hot

def mdn_get_most_probable_sigma_and_mu(log_pi, log_sigma, mu, lengths=None):
    dim_wise = len(log_pi.shape) == 4
    _, _, num_gaussians, _ = mu.shape
    # Get the indexes of the largest log_pi
    _, max_component = torch.max(log_pi, dim=2)  # (B, T) or (B, T, C_out)

    # Convert max_component to one_hot manner
    # if dim_wise: (B, T, D_out) -> (B, T, D_out, G)
    # else: (B, T) -> (B, T, G)
    one_hot = to_one_hot(max_component, num_gaussians, lengths=lengths)

    if dim_wise:
        # (B, T, G, D_out)
        one_hot = one_hot.transpose(2, 3)
        assert one_hot.shape == mu.shape
    else:
        # Expand the dim of one_hot as  (B, T, G) -> (B, T, G, d_out)
        one_hot = one_hot.unsqueeze(3).expand_as(mu)

    # Multiply one_hot and sum to get mean(mu) and standard deviation(sigma)
    # of the Gaussians whose weight coefficient(log_pi) is the largest.
    #  (B, T, G, d_out) -> (B, T, d_out)
    max_mu = torch.sum(mu * one_hot, dim=2)
    max_sigma = torch.exp(torch.sum(log_sigma * one_hot, dim=2))

    return max_sigma, max_mu

class ModelWrapper(torch.nn.Module):
    def __init__(self,model):
        super(ModelWrapper,self).__init__()
        self.model=model
    
    def forward(self, x, lengths=None):
        log_pi, log_sigma, mu = self.model.forward(x, lengths)
        sigma, mu = mdn_get_most_probable_sigma_and_mu(log_pi, log_sigma, mu, lengths)
        return mu, sigma

def export_model(config:DictConfig, typ:str ,device:str="cpu"):
    #load model
    model_config = OmegaConf.load(to_absolute_path(config[typ].model_yaml))
    model = hydra.utils.instantiate(model_config.netG).to(device)
    pth_path = config[typ]["checkpoint"]
    onnx_path = pth_path[:-4]+".onnx"
    checkpoint = torch.load(config[typ].checkpoint,
        map_location=lambda storage,loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    
    model.eval()
    
    #dummy input
    question_path = to_absolute_path(config.question_path)
    binary_dict, numeric_dict = \
        hts.load_question_set(question_path, append_hat_for_LL=False)
    dim = len(binary_dict)+len(numeric_dict)
    dummy_input = torch.from_numpy(numpy.zeros((1,1,dim))).float().to(device)
    lengths = torch.from_numpy(numpy.array([dummy_input.shape[1]],dtype=numpy.int64))
    input_tuple = (dummy_input, lengths)
    
    if model.prediction_type() == PredictionType.PROBABILISTIC:
        global modeltype
        modeltype = type(model)
        model_wrapper = ModelWrapper(model)
        
        #export onnx
        torch.onnx.export(
            model_wrapper, 
            input_tuple,
            onnx_path, 
            input_names=["linguistic_features","lengths"], 
            output_names=["max_mu","max_sigma"], 
            dynamic_axes={
                'linguistic_features':{1:'n_phonemes'}, 
                'max_mu':{1:'n_phonemes'},
                'max_sigma':{1:'n_phonemes'}
                }  
            )
        
            
    else:
        torch.onnx.export(
            model, 
            input_tuple,
            onnx_path, 
            input_names=["linguistic_features","lengths"], 
            output_names=["result"], 
            dynamic_axes={
                'linguistic_features':{1:'n_phonemes'}, 
                'result':{1:'n_phonemes'}
                }  
            )
        print(f'{datetime.now()} : exported {onnx_path}')

def export_minmaxscaler(path:str,encoding:str="utf8"):
    #export MinMaxScaler to json, which is c#-readable
    scaler=joblib.load(path)
    assert(type(scaler)==sklearn.preprocessing._data.MinMaxScaler)
    assert(scaler.feature_range==(0,1))
    scaler_output: List[Dict[str,float]] = [
        {
            "xmin":float(xmin),
            "scale":float(scale)
        } for (xmin,scale) in zip(scaler.data_min_,scaler.scale_)]
    output_path = path[:-7]+".json"
    with open(output_path,"w",encoding=encoding) as output_file:
        json.dump(scaler_output,output_file)
    print(f'{datetime.now()} : exported {output_path}')

def export_standardscaler(path:str,encoding:str="utf8"):
    scaler=joblib.load(path)
    assert(type(scaler)==sklearn.preprocessing._data.StandardScaler)
    scaler_output: List[Dict[str,float]] = [
        {
            "xmin":float(xmin),
            "scale":float(1/scale)
        } for (xmin,scale) in zip(scaler.mean_,scaler.scale_)]
    output_path = path[:-7]+".json"
    with open(output_path,"w",encoding=encoding) as output_file:
        json.dump(scaler_output,output_file)
    print(f'{datetime.now()} : exported {output_path}')

def export(voice_dir:str):
    # Load enuconfig
    path_enuconfig = os.path.join(voice_dir, 'enuconfig.yaml')
    if not os.path.exists(path_enuconfig):
        raise Exception('enuconfig.yaml not found')
    os.chdir(voice_dir)
    print(f'{datetime.now()} : reading enuconfig')
    config = DictConfig(OmegaConf.load(path_enuconfig))
    typs = ("timelag","duration")
    for typ in typs:
        set_checkpoint(config, typ)
        set_normalization_stat(config, typ)

    for typ in typs:
        #export scalers
        export_minmaxscaler(config[typ].in_scaler_path)
        export_standardscaler(config[typ].out_scaler_path)
        #export models
        export_model(config,typ)
    
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="voicebank folder to export",nargs="?", default='')
    args = parser.parse_args()
    print(args.path)
    if(args.path==""):
        print("please input the path of your ENUNU voicebank:")
        path=input()
    else:
        path=args.path
    export(path)   
    
if(__name__=="__main__"):
    main()