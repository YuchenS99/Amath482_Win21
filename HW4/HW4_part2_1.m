clc; clear all; close all;

% Part 1 - load in and reshape train and test data
[images_train, labels_train] = mnist_parse('train-images-idx3-ubyte',...
                               'train-labels-idx1-ubyte');
[images_test, labels_test] = mnist_parse('t10k-images-idx3-ubyte',...
                            't10k-labels-idx1-ubyte');
% cast the images to double
images_train = im2double(images_train);
images_test = im2double(images_test);
% reshape to one image per column for train data
images_train = reshape(images_train, [28*28, 60000]);
[M, N] = size(images_train);
train_data = images_train - repmat(mean(images_train,2),1,N);
% reshape to one image per column for test data
images_test = reshape(images_test, [28*28, 10000]);
[M, N] = size(images_test);
test_data = images_test - repmat(mean(images_test,2),1,N);

%% Part 2 - Q1: train data
% choose two digits 3 and 7
label3 = find(labels_train == 3);
digit3 = train_data(:,label3);
label7 = find(labels_train == 7);
digit7 = train_data(:,label7);
%%
% apply SVD to digit 3 and 7 data
[U,S,V] = svd([digit3, digit7], 'econ');
%%
% project on to PCA space
feature = 87;
n3 = size(digit3,2);
n7 = size(digit7,2);
digits = S*V';
digit3s = digits(1:feature,1:n3);
digit7s = digits(1:feature,n3+1:n3+n7);
%%
% Calculate scatter matrices
m3 = mean(digit3s,2);
m7 = mean(digit7s,2);
Sw = 0; % within class variances
for k = 1:n3
    Sw = Sw + (digit3s(:,k) - m3)*(digit3s(:,k) - m7)';
end
for k = 1:n7
   Sw =  Sw + (digit7s(:,k) - m7)*(digit7s(:,k) - m7)';
end
Sb = (m3-m7)*(m3-m7)'; % between class

% Find the best projection line
[V2, D] = eig(Sb,Sw); % linear disciminant analysis
[lambda, ind] = max(abs(diag(D)));
w = V2(:,ind);
w = w/norm(w,2);

% project on to w
vdigit3 = w'*digit3s;
vdigit7 = w'*digit7s;

% Make digit 3 below the threshold
if mean(vdigit3) > mean(vdigit7)
    w = -w;
    vdigit3 = -vdigit3;
    vdigit7 = -vdigit7;
end

% Find the threshold value
sortd3 = sort(vdigit3);
sortd7 = sort(vdigit7);

t1 = length(sortd3);
t2 = 1;
while sortd3(t1) > sortd7(t2)
    t1 = t1 - 1;
    t2 = t2 + 1;
end
threshold = (sortd3(t1) + sortd7(t2))/2;
%% Part 2 - Q1: test data
% Find digit 3 and digit 7 in the test data
label3_test = find(labels_test == 3);
digit3_test = test_data(:,label3_test);
label7_test = find(labels_test == 7);
digit7_test = test_data(:,label7_test);
test37 = [digit3_test digit7_test];
%% Part 2 - Q1: test
testNum = size(test37, 2);
testMat = U'* test37;
pval = w'*testMat(1:87,:);
ResVec = (pval > threshold);
trueRes = [3*ones(1,1010) 7*ones(1,1028)];
trueRes = (trueRes == 7);
err = abs(ResVec - trueRes);
errNum = sum(err);
sucRate = 1 - errNum/testNum;
%%










