import os
import json
import yaml
import pickle
import librosa
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from importlib import import_module
from torch import nn, Tensor
from models.RawNet import RawNet
from models.RawGAT_models.model_RawGAT_ST_add import RawGAT_ST
from typing import Dict, List, Union
from sklearn.mixture import GaussianMixture

class ASD:
    
    def __init__(self, model_type='aasist', generate_score_file=False, save_path='./score_files/'):
        self.model_type = model_type
        self.gen_score_file = generate_score_file

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'                  
        print('Device: {}'.format(self.device))

        if self.gen_score_file:
            self.save_path = save_path
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)

        if self.model_type == 'aasist':
            config_path = './config/AASIST_ASVspoof5.conf'
            with open(config_path, "r") as f_json:
                config = json.loads(f_json.read())

            self.config = config
            self.model_config = config["model_config"]
            self.model = get_model(self.model_config, self.device)
            self.model.load_state_dict(torch.load(config["model_path"], map_location=self.device))

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
                audio_data, sr = librosa.load(cf_filename, sr=16000)
                audio_data_dict[cf.split('.')[0]] = audio_data

        score_df = self.produce_evaluation(audio_data_dict, speaker_name=speaker_name)
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
    module = import_module(f"models.{model_config['architecture']}")
    _model = module.Model(model_config)
    model = _model.to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("no. model params:{}".format(nb_params))
    return model

def pad(x: np.ndarray, max_len=64600) -> np.ndarray:
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x

class ASD_GMM:
    def __init__(self, features='cqcc', model_path='./models/weights', generate_score_file=False, save_path='./score_files/'):
        self.features = features
        self.gen_score_file = generate_score_file
        self.save_path = save_path
        self.model_path = model_path

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
        self.gmm_bona = GaussianMixture(covariance_type='diag')
        self.gmm_spoof = GaussianMixture(covariance_type='diag')
        
        if self.features == 'cqcc':
            bona_path = os.path.join(self.model_path, 'gmm_512_LA_cqcc', 'bonafide', 'gmm_final.pkl')
            spoof_path = os.path.join(self.model_path, 'gmm_512_LA_cqcc', 'spoof', 'gmm_final.pkl')
        elif self.features == 'lfcc':
            bona_path = os.path.join(self.model_path, 'gmm_512_LA_lfcc', 'bonafide', 'gmm_final.pkl')
            spoof_path = os.path.join(self.model_path, 'gmm_512_LA_lfcc', 'spoof', 'gmm_final.pkl')
        else:
            raise ValueError("Unsupported feature type")
        
        print(f"Loading Bonafide GMM model from {bona_path}")
        if not os.path.exists(bona_path):
            raise FileNotFoundError(f"Model file not found: {bona_path}")

        with open(bona_path, 'rb') as f:
            gmm_dict = pickle.load(f)
            self.gmm_bona._set_parameters(gmm_dict)

        print(f"Loading Spoof GMM model from {spoof_path}")
        if not os.path.exists(spoof_path):
            raise FileNotFoundError(f"Model file not found: {spoof_path}")

        with open(spoof_path, 'rb') as f:
            gmm_dict = pickle.load(f)
            self.gmm_spoof._set_parameters(gmm_dict)

    def extract_features(self, audio_data):
        y, sr = audio_data, 16000
        if self.features == 'cqcc':
            return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=60).T
        elif self.features == 'lfcc':
            return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=60).T
        else:
            raise ValueError("Unsupported feature type")

    def produce_evaluation(self, data_dict: Dict, speaker_name='Barack_Obama') -> pd.DataFrame:
        fname_list = []
        score_list = []

        for utt_id, audio_data in tqdm(data_dict.items()):
            features = self.extract_features(audio_data)
            bona_score = self.gmm_bona.score(features)
            spoof_score = self.gmm_spoof.score(features)
            score = bona_score - spoof_score

            fname_list.append(utt_id)
            score_list.append(score)

        score_dict = {'filename': fname_list, 'cm-score': score_list}
        score_df = pd.DataFrame(data=score_dict)

        if self.gen_score_file:
            score_file = os.path.join(self.save_path, f"{speaker_name}_{self.features}_out_scores.txt")
            score_df.to_csv(score_file, index=False, sep="\t")
            print(f"Scores saved to {score_file}")

        return score_df