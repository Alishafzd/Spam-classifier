%% Initialization
clear ; close all; clc

%% Create Dataset
dataCreate();

%% Train Linear SVM for Spam Classification

% Load the Spam Email dataset (X and y)
readtable('emailsTable.mat')

% Shuffle dataset
idx = randperm(size(X, 1));
X = X(idx, :);
y = y(idx, :);

% Split dataset into train and test
X_train = X(1:4000, :);
y_train = y(1:4000, :);

model = fitcsvm(X_train, y_train);

p = model.predict(X_train);

fprintf('Training Accuracy: %f\n', mean(double(p == y_train)) * 100);

%% Test Spam Classification 
%  After training the classifier, we can evaluate it on a test set.

% Use test set
X_test = X(4001:5730, :);
y_test = y(4001:5730, :);

fprintf('\nEvaluating the trained Linear SVM on a test set ...\n')

p = model.predict(X_test);

fprintf('Test Accuracy: %f\n', mean(double(p == y_test)) * 100);


%% Top Predictors of Spam
%  The following code finds the words with the highest weights in the 
%  classifier.

% Predict training set
p = model.predict(X_train);

% Find weight of each word
temp = X_train(p == 1, :);
weights = sum(temp, 1);

% Sort the weights and obtin the vocabulary list
[weight, idx] = sort(weights, 'descend');
vocabList = getVocabList();

fprintf('\nTop predictors of spam: \n');
for i = 1:15
    fprintf(' %-15s (%f) \n', vocabList{idx(i)}, weight(i));
end

fprintf('\n\n');


%% Try an Email

% Set the file to be read 
filename = 'spamSample1.txt';

% Read and predict
file_contents = readFile(filename);
word_indices  = processEmail(file_contents);
x             = emailFeatures(word_indices);
p = svmPredict(model, x);

fprintf('\nProcessed %s\n\nSpam Classification: %d\n', filename, p);
fprintf('(1 indicates spam, 0 indicates not spam)\n\n');

