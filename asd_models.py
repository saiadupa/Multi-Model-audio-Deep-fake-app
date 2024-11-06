import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from importlib import import_module
from typing import Dict, List, Union
from torch import Tensor, nn
import soundfile as sf
import librosa
import yaml

from models.RawNet import RawNet
from models.RawGAT_models.model_RawGAT_ST_add import RawGAT_ST

class ASD:
    
    def __init__(self, model_type='aasist', generate_score_file=False, save_path = './score_files/'):

        self.model_type = model_type
        self.gen_score_file = generate_score_file

        # GPU device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'                  
        print('Device: {}'.format(self.device))

        if self.gen_score_file:
            self.save_path = save_path
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)

        # Initialize AASIST model
        if self.model_type == 'aasist':
            config_path = './config/AASIST_ASVspoof5.conf'

            with open(config_path, "r") as f_json:
                config = json.loads(f_json.read())

            self.config = config
            self.model_config = config["model_config"]
            self.model = get_model(self.model_config, self.device)
            self.model.load_state_dict(torch.load(config["model_path"], map_location=self.device))

        # Initialize RawNet model
        elif self.model_type == 'rawnet':
            dir_yaml = os.path.splitext('./config/model_config_RawNet')[0] + '.yaml'

            with open(dir_yaml, 'r') as f_yaml:
                parser1 = yaml.load(f_yaml, yaml.Loader)
                model = RawNet(parser1['model'], self.device)
                nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
                self.model = model.to(self.device)
                print("no. model params:{}".format(nb_params))

            model_path = './models/weights/RawNet2/RawNet2_best_model_laundered_train.pth'

            if model_path:
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                print('Model loaded : {}'.format(model_path))

        # Initialize RawGAT model
        elif self.model_type == 'rawgat':
            dir_yaml = os.path.splitext('./config/model_config_RawGAT_ST')[0] + '.yaml'

            with open(dir_yaml, 'r') as f_yaml:
                parser = yaml.load(f_yaml, yaml.Loader)

            self.model = RawGAT_ST(parser['model'], self.device)
            model_path = './models/weights/RawGAT/RawGAT_ST_add/Best_epoch.pth'

            if model_path:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print('Model loaded : {}'.format(model_path))

        else:
            raise ValueError("Unsupported model type: {}".format(self.model_type))

    
    def run(self, audio_data_dict: Dict, use_saved_chunks=False, chunk_dir='./chunks', speaker_name='Barack_Obama'):

        if use_saved_chunks:
            chunk_files = os.listdir(chunk_dir)
            audio_data_dict = {}
            for cf in chunk_files:
                cf_filename = os.path.join(chunk_dir, cf)
                audio_data, fs = librosa.load(cf_filename, sr=16000)
                audio_data_dict[cf.split('.')[0]] = audio_data

        score_df = self.produce_evaluation(data_dict=audio_data_dict, speaker_name=speaker_name)
        return score_df


    def produce_evaluation(self, data_dict: Dict, speaker_name='Barack_Obama') -> pd.DataFrame:
        
        model = self.model
        model.eval()

        fname_list = []
        score_list = []
        for utt_id, audio_data in tqdm(data_dict.items()):
            X_pad = pad(audio_data, 64600)
            x_inp = Tensor(X_pad)
            x_inp = x_inp.unsqueeze(0).float().to(self.device)

            with torch.no_grad():
                if self.model_type == 'aasist':
                    _, audio_out = model(x_inp)
                else:
                    audio_out = model(x_inp)

                audio_score = audio_out[:, 1].data.cpu().numpy().ravel()
            
            fname_list.append(utt_id)
            score_list.extend(audio_score.tolist())

        score_dict = {'filename': fname_list, 'cm-score': score_list}
        score_df = pd.DataFrame(data=score_dict)

        if self.gen_score_file:
            score_file = os.path.join(self.save_path, f"{speaker_name}_{self.model_type}_out_scores.txt")
            score_df.to_csv(score_file, index=False, sep="\t")
            print(f"Scores saved to {score_file}")

        return score_df


def get_model(model_config: Dict, device: str) -> nn.Module:
    """Define DNN model architecture"""
    module = import_module(f"models.{model_config['architecture']}")
    _model = module.Model(model_config)
    model = _model.to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("no. model params:{}".format(nb_params))
    return model

    
def pad(x: np.ndarray, max_len=64600) -> np.ndarray:
    """Pad the input array to the specified length"""
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x