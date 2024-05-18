#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SELFIES: a robust representation of semantically constrained graphs with an
    example application in chemistry (https://arxiv.org/abs/1905.13741)
    by Mario Krenn, Florian Haese, AkshatKuman Nigam, Pascal Friederich,
    Alan Aspuru-Guzik.

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

    VAEEncoder
        fully connected, 3 layer neural network - encodes a one-hot
        representation of molecule (in SMILES or SELFIES representation)
        to latent space

    VAEDecoder
        decodes point in latent space using an RNN

    latent_space_quality
        samples points from latent space, decodes them into molecules,
        calculates chemical validity (using RDKit's MolFromSmiles), calculates
        diversity
"""

import os
import numpy as np
import torch
from torch import nn
import pickle as pkl
import yaml
from scipy.spatial import distance
from rdkit import rdBase
from models.chem_vae_benchmark import RNN_decoder, MLP_encoder, train_model, is_correct_smiles, generate_samples_latent, reconstruct, compute_recon_quality
from rdkit import Chem, DataStructs 
from rdkit.Chem import Draw, AllChem
from utils.one_hot_encoding import smile_to_hot, selfies_to_hot, multiple_selfies_to_int
from sklearn.metrics import jaccard_score
import selfies as sf
import matplotlib.pylab as plt
from utils.one_hot_encoding import \
    multiple_selfies_to_hot, multiple_smile_to_hot, get_selfie_and_smiles_encodings_for_dataset

rdBase.DisableLog('rdApp.error')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_all_recons_for_mol(pin_mol, encoding_alphabet, num_samples, vae_encoder, vae_decoder, type_of_encoding):

    '''Sample num_samples times the encoding distribution of a pin molecule'''

    int_to_char = dict((i, c) for i, c in enumerate(encoding_alphabet))
        
    # mol_idx = np.random.randint(0,len(data_test))
    # test_mol = data_test[mol_idx].reshape(1,21,18)
    # int_label = int_test[mol_idx].numpy()
    # mol_ids.append(mol_idx)
    
    # selfie = sf.encoding_to_selfies(int_label, int_to_char, 'label')
    # smile = sf.decoder(selfie)
                
    for i in range(len(num_samples)):
                
        ## Constructing a distribution of reconstructions
        
        z_samples, mu, sigma = generate_samples_latent(pin_mol, 50, vae_encoder)
        all_mols, selfie_mols, smiles_mols, one_hot_recons = reconstuct(z_samples, None, vae_decoder, 21, encoding_alphabet, type_of_encoding)   
        
        unique_smiles.append(np.unique(smiles_mols))
        
        test_mol_repeat = test_mol.repeat_interleave(len(z_samples),dim=0)
        
        avg_recon = compute_recon_quality(test_mol_repeat, one_hot_recons).item()
        
        print('Avg. Recon error: ' + str(avg_recon))
        print('Std: ' + str(sigma))
        
        return np.unique(smiles_mols), np.unique(selfie_mols)
               

def plot_epsilon_radius_viz(pin_mol_smiles, vae_encoder, vae_decoder, largest_molecule_len, encoding_alphabet, type_of_encoding):
    
    #pin_mol = 'NNC1OCOC1=NO'
    
    ms = [Chem.MolFromSmiles(pin_mol)]
    Draw.MolsToGridImage(ms,molsPerRow=1,subImgSize=(200,200),legends=[pin_mol])
    
    pin_selfie = sf.encoder(pin_mol)
    #int_encoded, pin_one_hot = smile_to_hot(pin_mol, largest_smiles_len, smiles_alphabet)
    int_encoded, pin_one_hot = selfies_to_hot(pin_selfie, largest_molecule_len, encoding_alphabet)
    pin_one_hot = torch.Tensor(pin_one_hot).to(device)
    pin_one_hot = pin_one_hot.reshape([1,21,18])
    
    pin_latent, pin_mu, pin_sd = vae_encoder.forward(pin_one_hot.flatten(start_dim=1))
    gauss_mvt_01 = torch.distributions.MultivariateNormal(loc=pin_mu.cpu(), covariance_matrix=0.1*torch.eye(50))
    gauss_mvt_05 = torch.distributions.MultivariateNormal(loc=pin_mu.cpu(), covariance_matrix=0.5*torch.eye(50))
    gauss_mvt_1 = torch.distributions.MultivariateNormal(loc=pin_mu.cpu(), covariance_matrix=1*torch.eye(50))
    
    epsilon_samples_01 = gauss_mvt_01.sample_n(100)
    epsilon_samples_05 = gauss_mvt_05.sample_n(100)
    epsilon_samples_1 = gauss_mvt_1.sample_n(100)
    
    all_mols, selfie_mols, smiles_mols_01, one_hot_recons_01 = reconstuct(epsilon_samples_01.squeeze(), None, vae_decoder, largest_molecule_len, encoding_alphabet, type_of_encoding)   
    all_mols, selfie_mols, smiles_mols_05, one_hot_recons_05 = reconstuct(epsilon_samples_05.squeeze(), None, vae_decoder, largest_molecule_len, encoding_alphabet, type_of_encoding)   
    all_mols, selfie_mols, smiles_mols_1, one_hot_recons_1 = reconstuct(epsilon_samples_1.squeeze(), None, vae_decoder, largest_molecule_len, encoding_alphabet, type_of_encoding)   
    
    ms = [Chem.MolFromSmiles(x) for x in smiles_mols_01[0:50]]
    a=Draw.MolsToGridImage(ms,molsPerRow=10,subImgSize=(200,200),legends=smiles_mols_01[0:50])
    
    ms = [Chem.MolFromSmiles(x) for x in smiles_mols_05[0:50]]
    b=Draw.MolsToGridImage(ms,molsPerRow=10,subImgSize=(200,200),legends=smiles_mols_05[0:50])
    
    ms = [Chem.MolFromSmiles(x) for x in smiles_mols_1[0:50]]
    c=Draw.MolsToGridImage(ms,molsPerRow=10,subImgSize=(200,200),legends=smiles_mols_1[0:50])
        
    js_01 = [jaccard_score(pin_one_hot.squeeze().cpu().numpy().flatten(), x.cpu().numpy().flatten()) for x in one_hot_recons_01]
    js_05 = [jaccard_score(pin_one_hot.squeeze().cpu().numpy().flatten(), x.cpu().numpy().flatten()) for x in one_hot_recons_05]
    js_1 = [jaccard_score(pin_one_hot.squeeze().cpu().numpy().flatten(), x.cpu().numpy().flatten()) for x in one_hot_recons_1]
    
    plt.figure()
    plt.title('Tanimoto kernel in an epsilon-neighbourhood (100 molecules)')
    plt.plot(np.sort(js_01), 'bo', label='epsilon=0.1')
    plt.plot(np.sort(js_05), 'go', label='epsilon=0.5')
    plt.plot(np.sort(js_1), 'ro', label='epsilon=1')
    plt.legend()
    plt.ylabel('Tanimoto')
    
    return a,b,c

def plot_neighbourhood_viz(vae_encoder, vae_decoder, pin_mol_smiles, latent_size):
    
    x = np.random.randn(latent_size)
    x /= np.linalg.norm(x)
    
    y = np.random.randn(latent_size)
    y -= y.dot(x) * x
    y /= np.linalg.norm(y)
    
    #z0 = "CN1C(C2=CC(NC3C[C@H](C)C[C@@H](C)C3)=CN=C2)=NN=C1"
    #pin_mol = "NNC1OCOC1=NO"
    
    pin_selfie = sf.encoder(pin_mol_smiles)
    int_encoded, pin_one_hot = selfies_to_hot(pin_selfie, largest_molecule_len, encoding_alphabet)
    pin_one_hot = torch.Tensor(pin_one_hot).to(device)
    pin_one_hot = pin_one_hot.reshape([1,21,18])
    vae_encoder = vae_encoder.cuda()
    vae_decoder = vae_decoder.cuda()
    
    pin_chem = Chem.MolFromSmiles(pin_mol_smiles)
  
    pin_latent, pin_mu, pin_sd = vae_encoder.forward(pin_one_hot.flatten(start_dim=1))
    z0 = pin_latent.cpu().detach().numpy()
    
    valid_mols_smiles_unique_label = []
    delta = 1
    nei_mols = []
    for dx in range(-3,3):
        for dy in range(-3,3):
            z = z0 + x * delta * dx + y * delta * dy
            z = torch.Tensor(z)
            all_mols, selfie, smile, one_hot = reconstuct(z, None, vae_decoder, 21, encoding_alphabet, type_of_encoding)
            #js = jaccard_score(pin_one_hot.squeeze().cpu().numpy().flatten(), one_hot.cpu().numpy().flatten())
            m = Chem.MolFromSmiles(smile[0])
            fp = AllChem.GetMorganFingerprint(m, 2)
            fp0 = AllChem.GetMorganFingerprint(pin_chem, 2)
            sim = DataStructs.TanimotoSimilarity(fp, fp0)
               # s = s + ' {:.2f}'.format(sim)
               # if s == smile0:
               #     s = '***['+s+']***'
            s = ' {:.2f}'.format(sim)
            valid_mols_smiles_unique_label.append(smile[0] + '\n' + '\n'  + s)
            nei_mols.append(smile)
    
    nei_labels = [s[0] for s in nei_mols]
    nei_mols = [Chem.MolFromSmiles(s[0]) for s in nei_mols]
    p = Draw.MolsToGridImage(nei_mols, molsPerRow=5, subImgSize=(200,200), legends=valid_mols_smiles_unique_label)
    return p

def sample_from_prior(vae_encoder, vae_decoder, num_mols, largest_molecule_len):

    num_mols = 500
    all_mols = np.empty(shape=(n,largest_molecule_len), dtype='int')

    for i in range(n):

        gathered_atoms = []
    
        hidden = vae_decoder.init_hidden().to(device)
    
        fancy_point = torch.randn(1, 1, vae_encoder.latent_dimension,
                                       device=device)
        # runs over letters from molecules (len=size of largest molecule)
        for _ in range(largest_molecule_len):
            
            out_one_hot, hidden = vae_decoder(fancy_point, hidden)
            out_one_hot = out_one_hot.flatten().detach()
            soft = nn.Softmax(0)
            out_one_hot = soft(out_one_hot)
    
            out_index = out_one_hot.argmax(0)
            gathered_atoms.append(out_index.data.tolist())
        
        all_mols[i] = gathered_atoms
    
    selfie_mols = []
    smile_mols = []

    for _ in range(n):
    
        molecule_pre = ''
        
        for i in all_mols[_]:
            molecule_pre += encoding_alphabet[i]
        molecule = molecule_pre.replace(' ', '')
        #molecule = molecule.replace('[nop]', '')
        selfie_mols.append(molecule)
        
        if type_of_encoding == 1:  # if SELFIES, decode to SMILES
            smiles_molecule = sf.decoder(molecule)
        smile_mols.append(smiles_molecule)
        
    return smile_mols, selfie_mols

def plot_smiles_grid(smile_mols, truncate_idx):

    valid = [is_correct_smiles(x) for x in smile_mols]
    valid_idx = [i for i, x in enumerate(valid) if x]
    invalid_idx = [i for i, x in enumerate(valid) if not x]
    valid_mols = [smile_mols[i] for i in valid_idx]
    
    ms = [Chem.MolFromSmiles(x) for x in valid_mols[0:truncate_idx]]
    p = Draw.MolsToGridImage(ms,molsPerRow=10,subImgSize=(200,200),legends=valid_mols[0:truncate_idx])
    return p

def smiles_to_svg(input_smiles: str, svg_file_name:str,  size=(400, 200)):

    molecule= Chem.MolFromSmiles(input_smiles)
    drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
    drawer.DrawMolecule(molecule) 
    return drawer.FinishDrawing()
    #svg = drawer.GetDrawingText().replace('svg:','')
    #with open(svg_file_name, 'w') as f:
    #    f.write(svg)
    #return 

if __name__ == '__main__':

    if os.path.exists("utils/settings.yml"):
        settings = yaml.safe_load(open("utils/settings.yml", "r"))
    else:
        print("Expected a file settings.yml but didn't find it.")
    
    print('--> Acquiring data...')
    type_of_encoding = settings['data']['type_of_encoding']
    file_name_smiles = settings['data']['qm9_file']
    
    if type_of_encoding == 0:
        print('Representation: SMILES')
        _, _, _, encoding_list, encoding_alphabet, largest_molecule_len, properties = \
            get_selfie_and_smiles_encodings_for_dataset(file_name_smiles)
    
        print('--> Creating one-hot encoding...')
        data = multiple_smile_to_hot(encoding_list, largest_molecule_len,
                                     encoding_alphabet)
        print('Finished creating one-hot encoding.')
    
    elif type_of_encoding == 1:
        print('Representation: SELFIES')
        encoding_list, encoding_alphabet, largest_molecule_len, smiles_list, smiles_alphabet, largest_smiles_len, properties = \
            get_selfie_and_smiles_encodings_for_dataset(file_name_smiles)
    
        print('--> Creating one-hot encoding...')
        data = multiple_selfies_to_hot(encoding_list, largest_molecule_len,
                                       encoding_alphabet)
        int_data = multiple_selfies_to_int(encoding_list, largest_molecule_len, encoding_alphabet)
        print('Finished creating one-hot encoding.')
    
    else:
        print("type_of_encoding not in {0, 1}.")
        #return
    
    len_max_molec = data.shape[1]
    len_alphabet = data.shape[2] 
    len_max_mol_one_hot = len_max_molec * len_alphabet
    
    print(' ')
    print(f"Alphabet has {len_alphabet} letters, "
          f"largest molecule is {len_max_molec} letters.")
    
    data_parameters = settings['data']
    batch_size = data_parameters['batch_size']
    
    encoder_parameter = settings['encoder']
    decoder_parameter = settings['decoder']
    training_parameters = settings['training']
    
    vae_encoder = MLP_encoder(in_dimension=len_max_mol_one_hot,
                             **encoder_parameter).to(device)
    vae_decoder = RNN_decoder(**decoder_parameter,
                             out_dimension=len(encoding_alphabet)).to(device)
    
    print('*' * 15, ': -->', device)
    
    data = torch.tensor(data, dtype=torch.float).to(device)
    
    train_valid_test_size = [0.5, 0.4, 0.1]
    #data = data[torch.randperm(data.size()[0])]
    idx_train_val = int(len(data) * train_valid_test_size[0])
    idx_val_test = idx_train_val + int(len(data) * train_valid_test_size[1])
    
    data_train = data[0:idx_train_val]
    data_valid = data[idx_train_val:idx_val_test]
    data_test = data[idx_val_test:]
    int_valid = int_data[idx_train_val:idx_val_test]
    int_test = int_data[idx_val_test:]
    
    print("start training")
    train_model(**training_parameters,
                vae_encoder=vae_encoder,
                vae_decoder=vae_decoder,
                batch_size=batch_size,
                data_train=data_train,
                data_valid=data_valid,
                alphabet=encoding_alphabet,
                type_of_encoding=type_of_encoding,
                sample_len=len_max_molec)
    

    ###### Save model state dictionary

    # name = 'qm9_model.pkl'
    # if os.path.isfile('trained_models/' + name):
    #       with open('trained_models/' + name, 'rb') as file:
    #           vae_encoder_sd, vae_decoder_sd = pkl.load(file)
    #           vae_encoder.load_state_dict(vae_encoder_sd)
    #           vae_decoder.load_state_dict(vae_decoder_sd)

    # with open('trained_models/' + name, 'wb') as file:
    #       pkl.dump((vae_encoder.cpu().state_dict(), vae_decoder.cpu().state_dict()), file)
 
    #########------Evaluation-------############
    
    ######## Block reconstruction test ##########
    
    all_mols, selfie_mols, smiles_mols, one_hot_recons = reconstruct(data_test, vae_encoder, vae_decoder, largest_molecule_len, encoding_alphabet, type_of_encoding)
    compute_recon_quality(data_test, one_hot_recons).item()
    
    ######### Sample from prior and decode to check validity ##########
    
    num_mols = 100
    smiles_mols, selfie_mols = sample_from_prior(vae_encoder, vae_decoder, num_mols, largest_molecule_len)
    plot_smiles_grid(smiles_mols, truncate_idx)

    ######## Neighbourhood viz on a grid defined by 2 orth. vectors ##########
    
    pin_mol_smiles = smiles_list[7777]
    latent_size = settings['decoder']['latent_dimension']
    neighbourhood_viz(vae_encoder, vae_decoder, pin_mol_smiles, latent_size)
    
    ####### epsilon radius visualiations ########

    a,b,c = plot_epsilon_radius_viz(pin_mol, vae_encoder, vae_decoder, largest_molecule_len, encoding_alphabet, type_of_encoding)
    
    ############# Return all reconstructions for specific mol ##########
    
    u_smiles, u_selfies = get_all_recons_for_mol(pin_mol, encoding_alphabet, num_samples, vae_encoder, vae_decoder, type_of_encoding)
    
    ######## Clean-up from this point on #########
    
    
    ## Drawing samples from latent space for 5 test molecules and plotting with ground truth
    

    def plot_all_recons_for_smile(pin_mol, unique_recons):
        
        ms = [Chem.MolFromSmiles(x) for x in u_smiles]
        
        legends = ['Samples']*len(ms) + ['Ground truth']
        ms.append(Chem.MolFromSmiles(pin_mol))
    
        p = Draw.MolsToGridImage(ms, legends=legends)
        p.show()
        
        
##############################

data_label = 'test' ## 'test'

plt.figure()
#plt.scatter(mu_plot[:,col_id1], mu_plot[:,col_id2], c='b', s=3, label='Avg sigma ' + str(np.mean(sigma_plot)), cmap='jet')
plt.scatter(mu_plot[:,col_id1], mu_plot[:,col_id2], c=js, marker='x', label='Avg sigma ' + str(np.mean(sigma_plot)), cmap='jet_r')
#plt.scatter(plane_samples[:,col_id1], plane_samples[:,col_id2], c='r',s=2)
plt.errorbar(mu_plot[:,col_id1], mu_plot[:,col_id2], yerr=3*sigma_plot[:,col_id2], fmt='None', color='k', alpha=0.2, errorevery=1)
plt.errorbar(mu_plot[:,col_id1], mu_plot[:,col_id2], xerr=3*sigma_plot[:,col_id1], fmt='None', color='k', alpha=0.4, errorevery=1)
plt.ylabel(r'Latent dim 1', fontsize='large')
plt.xlabel(r'Latent dim 2', fontsize='large')
plt.title(r'2d slice of latent spacw w. 3$\sigma$ uncertainty ' + '[Data: ' + data_label + ']')
plt.legend()
plt.savefig(fname=data_label + ' 2d_latent_slice')
cbar = plt.colorbar()
cbar.set_label('tanimoto kernel score', rotation=270)

interp_mols = torch.stack((mols[0], mols[1])).reshape(2,21,18)
z, mus, log_vars = vae_encoder.forward(interp_mols.flatten(start_dim=1))

mu_plot = mus.cpu().detach().numpy()
sigma_plot = torch.sqrt(torch.exp(log_vars)).cpu().detach().numpy()
plt.scatter(mu_plot[:,col_id1], mu_plot[:,col_id2], c='m', marker='^', label='Avg sigma ' + str(np.mean(sigma_plot)), cmap='jet_r')
plt.plot(mu_plot[:,col_id1], mu_plot[:,col_id2], c='r', linestyle='--', lw=2)


    #### Highlight latent space by precise and incorrect reconstructions

    all_mols, selfie_mols, smiles_mols, one_hot_recons = reconstuct(test_sample, vae_encoder, vae_decoder, 21, encoding_alphabet, type_of_encoding)


    boolean = []
    js = []
    for i in range(one_hot_recons.shape[0]): 
        boolean.append(torch.equal(test_sample[i], one_hot_recons[i]))
        js.append(jaccard_score(test_sample[i].cpu().flatten(), one_hot_recons[i].cpu().flatten()))
        
    matches = [1 if x is True else 2 for x in boolean]
        

    ### Extract points from a (d-1) dimensional subspace
    
    normal = torch.distributions.MultivariateNormal(loc=torch.Tensor([0]*49), covariance_matrix=0.1*torch.eye(49))
    
    random_d_vector_direction = torch.Tensor([1]*50)
    
    samples = normal.sample((1000,))
    
    dth_coordinate = -torch.matmul(random_d_vector_direction[0:49], samples.T)
    
    plane_samples = torch.cat((samples, dth_coordinate.unsqueeze(1)),axis=-1)
    
    def latent_space_quality(plane_samples, vae_encoder, vae_decoder, type_of_encoding,
                             alphabet, sample_num, sample_len):
        total_correct = 0
        all_correct_molecules = set()
        print(f"latent_space_quality:"
              f" Take {sample_num} samples from the latent space")
        gt_smiles = []
    
        for _ in range(sample_num):
    
            molecule_pre = ''
            
            for i in all_mols[_]:
                molecule_pre += alphabet[i]
            molecule = molecule_pre.replace(' ', '')
    
            if type_of_encoding == 1:  # if SELFIES, decode to SMILES
                molecule = sf.decoder(molecule)
                gt_smiles.append(molecule)
    
            if is_correct_smiles(molecule):
                total_correct += 1
                all_correct_molecules.add(molecule)
    
        return total_correct, len(all_correct_molecules)

    #### Linearly interpolate between two samples in 50d latent space and visualise the molecules
    
    mols = []
    idx = []
    for _ in [1,2]:
        mol_idx = np.random.randint(0,len(data_valid))
        test_mol = data_valid[mol_idx].reshape(1,21,18)
        mols.append(test_mol)
        idx.append(mol_idx)
        
    z_1 = vae_encoder.forward(mols[0].flatten(start_dim=1))[0]
    z_2 = vae_encoder.forward(mols[1].flatten(start_dim=1))[0]
    
    latents = torch.stack([z_1 + (z_2 - z_1)*t for t in np.linspace(0, 1, 10)])
    dim = vae_encoder.latent_dimension
    
    for i in range(10):
    
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
    
    int_list, one_hot_recons = multiple_selfies_to_hot(selfie_mols, largest_molecule_len, encoding_alphabet)
        
    ms = [Chem.MolFromSmiles(x) for x in smile_mols]
    Draw.MolsToGridImage(ms, legends=smile_mols)


    plt.figure(figsize=(10,5))
    plt.matshow(one_hot_recons[0:500].flatten(start_dim=1).cpu().detach())
    #plt.title('Ground Truth')
    plt.title('Reconstructions')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('One-hot string embedding')
    plt.ylabel('Molecules')