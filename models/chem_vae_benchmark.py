#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Variational Autoencoder (VAE) for chemistry
        comparing SMILES and SELFIES representation using reconstruction
        quality, diversity and latent space validity as metrics of
        interest
        
    information:
        ML framework: pytorch
        chemistry framework: RDKit

        get_selfie_and_smiles_encodings_for_dataset
            generate complete encoding (inclusive alphabet) for SMILES and
            SELFIES given a data file

        MLP_encoder
            fully connected, 3 layer neural network - encodes a one-hot
            representation of molecule (in SMILES or SELFIES representation)
            to latent space

        RNN_decoder
            decodes point in latent space using an RNN

        latent_space_quality
            samples points from latent space, decodes them into molecules,
            calculates chemical validity (using RDKit's MolFromSmiles), calculates
            diversity
            
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import yaml
from rdkit import rdBase
from rdkit.Chem import MolFromSmiles
from torch import nn

import selfies as sf
from utils.one_hot_encoding import \
    multiple_selfies_to_hot, multiple_smile_to_hot, get_selfie_and_smiles_encodings_for_dataset
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLP_encoder(nn.Module):

    def __init__(self, in_dimension, layer_1d, layer_2d, layer_3d,
                 latent_dimension):
        """
        Fully Connected layers to encode molecule to latent space
        """
        super(MLP_encoder, self).__init__()
        self.latent_dimension = latent_dimension

        # Reduce dimension up to second last layer of Encoder
        self.encode_nn = nn.Sequential(
            nn.Linear(in_dimension, layer_1d),
            nn.ReLU(),
            nn.Linear(layer_1d, layer_2d),
            nn.ReLU(),
            nn.Linear(layer_2d, layer_3d),
            nn.ReLU()
        )

        # Latent space mean
        self.encode_mu = nn.Linear(layer_3d, latent_dimension)

        # Latent space variance
        self.encode_log_var = nn.Linear(layer_3d, latent_dimension)

    @staticmethod
    def reparameterize(mu, log_var):
        """
        This trick is explained well here:
            https://stats.stackexchange.com/a/16338
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        """
        Pass throught the Encoder
        """
        # Get results of encoder network
        h1 = self.encode_nn(x)

        # latent space
        mu = self.encode_mu(h1)
        log_var = self.encode_log_var(h1)

        # Reparameterize
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var
    
class RNN_decoder(nn.Module):

    def __init__(self, latent_dimension, gru_stack_size, gru_neurons_num,
                 out_dimension):
        """
        Through Decoder
        """
        super(RNN_decoder, self).__init__()
        self.latent_dimension = latent_dimension
        self.gru_stack_size = gru_stack_size
        self.gru_neurons_num = gru_neurons_num

        # Simple Decoder
        self.decode_RNN = nn.GRU(
            input_size=latent_dimension,
            hidden_size=gru_neurons_num,
            num_layers=gru_stack_size,
            batch_first=False)

        self.decode_FC = nn.Sequential(
            nn.Linear(gru_neurons_num, out_dimension),
        )

    def init_hidden(self, batch_size=1):
        weight = next(self.parameters())
        return weight.new_zeros(self.gru_stack_size, batch_size,
                                self.gru_neurons_num)

    def forward(self, z, hidden):
        
        """
        A forward pass throught the entire model.
        
        """
        # Decode
        l1, hidden = self.decode_RNN(z, hidden)
        decoded = self.decode_FC(l1)  # fully connected layer

        return decoded, hidden
    
def sample_latent_space(vae_encoder, vae_decoder, sample_len):
    
    vae_encoder.eval()
    vae_decoder.eval()

    gathered_atoms = []

    fancy_latent_point = torch.randn(1, 1, vae_encoder.latent_dimension,
                                     device=device)
    hidden = vae_decoder.init_hidden()

    # runs over letters from molecules (len=size of largest molecule)
    for _ in range(sample_len):
        
        out_one_hot, hidden = vae_decoder(fancy_latent_point, hidden)
        out_one_hot = out_one_hot.flatten().detach()
        soft = nn.Softmax(0)
        out_one_hot = soft(out_one_hot)

        out_index = out_one_hot.argmax(0)
        gathered_atoms.append(out_index.data.cpu().tolist())
    
    vae_encoder.train()
    vae_decoder.train()
    
    return gathered_atoms


def latent_space_quality(vae_encoder, vae_decoder, type_of_encoding,
                         alphabet, sample_num, sample_len):
    total_correct = 0
    all_correct_molecules = set()
    print(f"latent_space_quality:"
          f" Take {sample_num} samples from the latent space")

    for _ in range(1, sample_num + 1):

        molecule_pre = ''
        for i in sample_latent_space(vae_encoder, vae_decoder, sample_len):
            molecule_pre += alphabet[i]
        molecule = molecule_pre.replace(' ', '')

        if type_of_encoding == 1:  # if SELFIES, decode to SMILES
            molecule = sf.decoder(molecule)

        if is_correct_smiles(molecule):
            total_correct += 1
            all_correct_molecules.add(molecule)

    return total_correct, len(all_correct_molecules)

def quality_in_valid_set(vae_encoder, vae_decoder, data_valid, batch_size):
    
    data_valid = data_valid[torch.randperm(data_valid.size()[0])]  # shuffle
    num_batches_valid = len(data_valid) // batch_size

    quality_list = []
    
    for batch_iteration in range(min(25, num_batches_valid)):

        # get batch
        start_idx = batch_iteration * batch_size
        stop_idx = (batch_iteration + 1) * batch_size
        batch = data_valid[start_idx: stop_idx]
        _, trg_len, _ = batch.size()

        inp_flat_one_hot = batch.flatten(start_dim=1)
        latent_points, mus, log_vars = vae_encoder(inp_flat_one_hot)

        latent_points = latent_points.unsqueeze(0)
        hidden = vae_decoder.init_hidden(batch_size=batch_size)
        out_one_hot = torch.zeros_like(batch, device=device)
        for seq_index in range(trg_len):
            out_one_hot_line, hidden = vae_decoder(latent_points, hidden)
            out_one_hot[:, seq_index, :] = out_one_hot_line[0]

        # assess reconstruction quality
        quality = compute_recon_quality(batch, out_one_hot)
        quality_list.append(quality)

    return np.mean(quality_list).item()

def train_model(vae_encoder, vae_decoder,
                data_train, data_valid, num_epochs, batch_size,
                lr_enc, lr_dec, KLD_alpha,
                sample_num, sample_len, alphabet, type_of_encoding):
    """
    Train the Variational Auto-Encoder
    """

    print('num_epochs: ', num_epochs)

    # initialize an instance of the model
    optimizer_encoder = torch.optim.Adam(vae_encoder.parameters(), lr=lr_enc)
    optimizer_decoder = torch.optim.Adam(vae_decoder.parameters(), lr=lr_dec)

    data_train = data_train.clone().detach().to(device)
    num_batches_train = int(len(data_train) / batch_size)

    quality_valid_list = [0, 0, 0, 0]
    for epoch in range(num_epochs):

        data_train = data_train[torch.randperm(data_train.size()[0])]

        start = time.time()
        for batch_iteration in range(num_batches_train):  # batch iterator

            # manual batch iterations
            start_idx = batch_iteration * batch_size
            stop_idx = (batch_iteration + 1) * batch_size
            batch = data_train[start_idx: stop_idx]

            # reshaping for efficient parallelization
            inp_flat_one_hot = batch.flatten(start_dim=1)
            latent_points, mus, log_vars = vae_encoder(inp_flat_one_hot)

            # initialization hidden internal state of RNN (RNN has two inputs
            # and two outputs:)
            #    input: latent space & hidden state
            #    output: one-hot encoding of one character of molecule & hidden
            #    state the hidden state acts as the internal memory
            latent_points = latent_points.unsqueeze(0)
            hidden = vae_decoder.init_hidden(batch_size=batch_size)

            # decoding from RNN N times, where N is the length of the largest
            # molecule (all molecules are padded)
            out_one_hot = torch.zeros_like(batch, device=device)
            for seq_index in range(batch.shape[1]):
                out_one_hot_line, hidden = vae_decoder(latent_points, hidden)
                out_one_hot[:, seq_index, :] = out_one_hot_line[0]

            # compute ELBO
            loss = compute_elbo(batch, out_one_hot, mus, log_vars, KLD_alpha)

            # perform back propogation
            optimizer_encoder.zero_grad()
            optimizer_decoder.zero_grad()
            loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(vae_decoder.parameters(), 0.5)
            optimizer_encoder.step()
            optimizer_decoder.step()

            if batch_iteration % 30 == 0:
                end = time.time()

                # assess reconstruction quality
                quality_train = compute_recon_quality(batch, out_one_hot)
                quality_valid = quality_in_valid_set(vae_encoder, vae_decoder,
                                                     data_valid, batch_size)

                report = 'Epoch: %d,  Batch: %d / %d,\t(loss: %.4f\t| ' \
                         'quality: %.4f | quality_valid: %.4f)\t' \
                         'ELAPSED TIME: %.5f' \
                         % (epoch, batch_iteration, num_batches_train,
                            loss.item(), quality_train, quality_valid,
                            end - start)
                print(report)
                start = time.time()

        quality_valid = quality_in_valid_set(vae_encoder, vae_decoder,
                                             data_valid, batch_size)
        quality_valid_list.append(quality_valid)

        # only measure validity of reconstruction improved
        quality_increase = len(quality_valid_list) \
                           - np.argmax(quality_valid_list)
        if quality_increase == 1 and quality_valid_list[-1] > 50.:
            corr, unique = latent_space_quality(vae_encoder, vae_decoder,
                                                type_of_encoding, alphabet,
                                                sample_num, sample_len)
        else:
            corr, unique = -1., -1.

        report = 'Validity: %.5f %% | Diversity: %.5f %% | ' \
                 'Reconstruction: %.5f %%' \
                 % (corr * 100. / sample_num, unique * 100. / sample_num,
                    quality_valid)
        print(report)

        with open('results.dat', 'a') as content:
            content.write(report + '\n')

        if quality_valid_list[-1] < 70. and epoch > 200:
            break

        if quality_increase > 20:
            print('Early stopping criteria')
            break

def compute_elbo(x, x_hat, mus, log_vars, KLD_alpha):
    
    inp = x_hat.reshape(-1, x_hat.shape[2])
    target = x.reshape(-1, x.shape[2]).argmax(1)
    #target = x.reshape(-1, x.shape[2])

    criterion = torch.nn.CrossEntropyLoss()
    recon_loss = criterion(inp, target)
    kld = -0.5 * torch.mean(1. + log_vars - mus.pow(2) - log_vars.exp())

    return recon_loss + KLD_alpha * kld


def compute_recon_quality(x, x_hat):
    
    x_indices = x.reshape(-1, x.shape[2]).argmax(1)
    x_hat_indices = x_hat.reshape(-1, x_hat.shape[2]).argmax(1)
    
    #x_indices = x.reshape(-1, x.shape[2])
    #x_hat_indices = x_hat.reshape(-1, x_hat.shape[2])

    differences = 1. - torch.abs(x_hat_indices - x_indices)
    differences = torch.clamp(differences, min=0., max=1.).double()
    quality = 100. * torch.mean(differences)
    quality = quality.detach().cpu().numpy()

    return quality


##-------------------Evaluation functions------------------------------------###


def _make_dir(directory):
    os.makedirs(directory)

def save_models(encoder, decoder, epoch):
    out_dir = './saved_models/{}'.format(epoch)
    _make_dir(out_dir)
    torch.save(encoder, '{}/E'.format(out_dir))
    torch.save(decoder, '{}/D'.format(out_dir))

def generate_samples_latent(orig_mol, num_samples, vae_encoder):
    
    z_samples = torch.empty(size=(num_samples,vae_encoder.latent_dimension), device=device)
    z, mu, log_var = vae_encoder.forward(orig_mol.flatten(start_dim=1))
    dist = torch.distributions.Normal(loc=mu, scale=torch.sqrt(torch.exp(log_var))*2)

    for i  in range(num_samples):
        
        z_samples[i] =  dist.sample()
        
    return z_samples, mu, torch.sqrt(torch.exp(log_var))
    
def is_correct_smiles(smiles):
    """
    Using RDKit to calculate whether molecule is syntactically and
    semantically valid.
    """
    if smiles == "":
        return False

    try:
        return MolFromSmiles(smiles, sanitize=True) is not None
    except Exception:
        return False
    
def reconstruct(input_mols, vae_encoder, vae_decoder, largest_molecule_len, encoding_alphabet, type_of_encoding):

    vae_decoder.eval()
    
    if vae_encoder is not None:
        vae_encoder.eval()
        latents = vae_encoder.forward(input_mols.flatten(start_dim=1))[0]
        dim = vae_encoder.latent_dimension
    else:
        latents = input_mols.cuda()
        dim = input_mols.shape[1]

    n = len(input_mols)
    
    all_mols = np.empty(shape=(n,largest_molecule_len), dtype='int')
  
    for i in range(n):
    
        gathered_atoms = []
    
        hidden = vae_decoder.init_hidden()
    
        fancy_point = latents[i].reshape(1,1,dim)
        # runs over letters from molecules (len=size of largest molecule)
        for _ in range(largest_molecule_len):
            
            out_one_hot, hidden = vae_decoder(fancy_point, hidden)  
    
            out_one_hot = out_one_hot.flatten().detach()
            soft = nn.Softmax(0)
            out_one_hot = soft(out_one_hot)
    
            out_index = out_one_hot.argmax(0)
            gathered_atoms.append(out_index.data.cpu().tolist())
        
        all_mols[i] = gathered_atoms
        
    smile_mols = []
    selfie_mols = []
    
    for _ in range(n):

        molecule_pre = ''
        
        for i in all_mols[_]:
            molecule_pre += encoding_alphabet[i]
        molecule = molecule_pre.replace(' ', '')
        selfie_mols.append(molecule)

        if type_of_encoding == 1:  # if SELFIES, decode to SMILES
            smiles_molecule = sf.decoder(molecule)
            
        smile_mols.append(smiles_molecule)
    
    one_hot_recons = multiple_selfies_to_hot(selfie_mols, largest_molecule_len, encoding_alphabet)
    return all_mols, selfie_mols, smile_mols, torch.Tensor(one_hot_recons).to(device)

  
def continuous_to_atoms(latent_samples, vae_decoder, sample_len, largest_molecule_len):
    
    all_mols = np.empty(shape=(len(latent_samples),largest_molecule_len), dtype='int')

    for i in range(len(latent_samples)):
        
            gathered_atoms = []
            hidden = vae_decoder.init_hidden()
            fancy_point = latent_samples[i].reshape(1,1,50).cuda()
                
            # runs over letters from molecules (len=size of largest molecule)
            for _ in range(sample_len):
                
                out_one_hot, hidden = vae_decoder(fancy_point, hidden)
                out_one_hot = out_one_hot.flatten().detach()
                soft = nn.Softmax(0)
                out_one_hot = soft(out_one_hot)
        
                out_index = out_one_hot.argmax(0)
                gathered_atoms.append(out_index.data.cpu().tolist())
            
            all_mols[i] = gathered_atoms
     
        