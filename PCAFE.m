name={'isolet','sonar','Hill_Valley_without_noise_Training','Epileptic Seizure Recognitio','redwine'...
     'whitewine','MF','SPECTHeart','Statlog','Madelon','Libras Movement','LSVT_voice_rehabilitation','drivFaceD'...
     'Urban land cover','MEU-Mobile KSD 2016','ionosphere','ORL','COIL20','orlraws10P','pixraw10P','warpAR10P','warpPIE10P','Yale'...
     'GLIOMA','lung_discrete','colon','lung','ForestTypes'};
 addpath(genpath('dataset'));
num_dataset=length(name);
id=8; %:num_dataset   %choess the datasets whatever you want %[2,3,8,11,12,14,16,23,24,25,26,28]
dataset=name{id};
switch dataset    
    case 'isolet'
 load('isolet.mat')
    dn=5:5:50;
    dnsize=10;
    dnd=5;
    case 'sonar'
 load('sonar.mat')
    dn=5:5:50;
    dnsize=10;
    dnd=5;
    case 'Hill_Valley_without_noise_Training'
 load('Hill.mat')
    dn=5:5:50;
    dnsize=10;
    dnd=5;
    case 'Epileptic Seizure Recognitio'
 load('Epileptic Seizure Recognitio.mat')
    dn=5:5:50;
    dnsize=10;
    dnd=5;
    case 'redwine'
 load('redwine.mat')
    dn=1:1:10;
    dnsize=10;
    dnd=1;
     case 'whitewine'
 load('whitewine.mat')
    dn=1:1:10;
    dnsize=10;
    dnd=1;
    case 'MF'
 load('MF.mat')
    dn=5:5:50;
    dnsize=10;
    dnd=5;
    case 'SPECTHeart'
 load('SPECTHeart.mat')
    dn=2:2:20;
    dnsize=10;
    dnd=2;
     case 'Statlog'
 load('Statlog.mat')
    dn=2:2:20;
    dnsize=10;
    dnd=2;
    case 'Madelon'
 load('Madelon.mat')
    dn=5:5:50;
    dnsize=10;
    dnd=5;
    case 'Libras Movement'
 load('Libras Movement.mat')
    dn=5:5:50;
    dnsize=10;
    dnd=5;
    case 'LSVT_voice_rehabilitation'
 load('LSVT_voice_rehabilitation.mat')
    dn=5:5:50;
    dnsize=10;
    dnd=5;
    case 'drivFaceD'
 load('drivFaceD.mat')
    dn=5:5:50;
    dnsize=10;
    dnd=5;
   case 'Urban land cover'
 load('Urban land cover.mat')
    dn=5:5:50;
    dnsize=10;
    dnd=5; 
    case 'MEU-Mobile KSD 2016'
 load('MEU-Mobile KSD 2016.mat')
    dn=5:5:50;
    dnsize=10;
    dnd=5; 
    case 'ionosphere'
 load('ionosphere.mat')
    dn=2:2:20;
    dnsize=10;
    dnd=2;
     case 'ORL'
 load('ORL.mat')
    data=[X,Y];
    dn=5:5:50;
    dnsize=10;
    dnd=5;
     case 'COIL20'
 load('COIL20.mat')
    data=[X,Y];
    dn=5:5:50;
    dnsize=10;
    dnd=5;
     case 'orlraws10P'
 load('orlraws10P.mat')
    data=[X,Y];
    dn=5:5:50;
    dnsize=10;
    dnd=5;
     case 'pixraw10P'
 load('pixraw10P.mat')
    data=[X,Y];
    dn=5:5:50;
    dnsize=10;
    dnd=5;
     case 'warpAR10P'
 load('warpAR10P.mat')
    data=[X,Y];
    dn=5:5:50;
    dnsize=10;
    dnd=5;
     case 'warpPIE10P'
 load('warpPIE10P.mat')
    data=[X,Y];
    dn=5:5:50;
    dnsize=10;
    dnd=5;
     case 'Yale'
 load('Yale.mat')
    data=[X,Y];
    dn=5:5:50;
    dnsize=10;
    dnd=5;
     case 'GLIOMA'
 load('GLIOMA.mat')
    data=[X,Y];
    dn=5:5:50;
    dnsize=10;
    dnd=5;
     case 'lung_discrete'
 load('lung_discrete.mat')
    data=[X,Y];
    dn=5:5:50;
    dnsize=10;
    dnd=5;
     case 'colon'
 load('colon.mat')
    data=[X,Y];
    dn=5:5:50;
    dnsize=10;
    dnd=5;
    case 'lung'
 load('lung.mat')
    data=[X,Y];
    dn=5:5:50;
    dnsize=10;
    dnd=5;
    case 'ForestTypes'
 load('ForestTypes.mat')
    dn=2:2:20;
    dnsize=10;
    dnd=2;
end 
k=10;
group=data(:,end);
class=unique(data(:,end));
[ mu,sigma,coeff,score,latent ]= PCA1(data(:,1:end-1));
data1=[score,data(:,end)];
%% Category group
for i=1:length(class)
        sa=[];
        sa=data1((group==class(i)),:);
        [number_of_smile_samples,~] = size(sa); % Column-observation
        smile_subsample_segments1 = round(linspace(1,number_of_smile_samples,k+1)); % indices of subsample segmentation points    
        data_group{i}=sa;
        smile_subsample_segments{i}=smile_subsample_segments1;
end
%% classify
for i=1:k   
    data_ts=[];data_tr =[];
    for j=1:length(class)
      smile_subsample_segments1=smile_subsample_segments{j};
      sa=data_group{j};
      test= sa(smile_subsample_segments1(i):smile_subsample_segments1(i+1) , :); % current_test_smiles
      data_ts=[test;data_ts] ; %Training data
      train = sa;
      train(smile_subsample_segments1(i):smile_subsample_segments1(i+1),:) = [];
      data_tr =[train;data_tr];%test data
    end
    mdl = fitcknn(data_tr(:,1:end-1),data_tr(:,end),'NumNeighbors',4,'Standardize',1);%KNN-classifier,now k=4
    Ac1=predict(mdl,data_ts(:,1:end-1)); 
    Fit(i)=sum(Ac1~=data_ts(:,end))/size(data_ts,1);
end
    fitness=mean(Fit)
    %% a is Classification error rate,  data1 is new matrix after feature extractionn
    [a,b]=min(fitness)
%% PCA main function
function [ mu,sigma,coeff,score,latent ] = PCA1( originalData )
% originalData -- raw data, each row representing a set;
% standardized data: mu - the mean of each column; Sigma - the standard deviation of each column;
% PCA dimensionality reduction data: coeff-coefficient matrix; score-PCA dimensionality reduction results; latent-all principal components impact rate (%).
%Data standardization and preservation of the parameters used for standardization;
%PCA dimensionality reduction;
%Selection dimension (usually the first few dimensions with cumulative impact rate greater than 95%) and retain the sample mean and coefficient matrix used;
%1 Data standardization: Z-standardized data; mu-mean of each column; sigma-standard deviation of each column.
[Z,mu,sigma]=zscore(originalData);
%2 PCA dimensionality reduction;£º
[coeff,score,latent] = pca(Z);
%3¡¢Choice dimension
latent=100*latent/sum(latent);
A=length(latent);
percent_threshold=95;           %Percentage threshold is used to determine the number of principal components retained.
percents=0;                    
for n=1:A
    percents=percents+latent(n);
    if percents>percent_threshold
        break;
    end
end
coeff=coeff(:,1:n);               %The coefficient matrix that meets the requirement of cumulative impact rate of principal component;
score=score(:,1:n);               %<br>save PCA mu sigma coeff latent
end