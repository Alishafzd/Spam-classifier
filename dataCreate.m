function emailsTable = dataCreate()
%DATACREATE This function creates data set from .txt emails
%   The function reads all emails in the Extracted_contents folder, and
%   creates indice vector for each email and saves the results in X data
%   set, and saves the type of the email (spam/ham) in the y data set

% Load email.csv
emails = readtable('emails.csv');

% Load vocab list
vocabList = readtable('vocab.txt');

% Email types (spam/ham) dataset
y = emails{:, 2};
    
% Prealocate worde indices (X) and email type (y) matrix (spam: 1, ham: 0)
X = zeros(size(emails, 1), size(vocabList, 1));

for row = 1:size(emails, 1)
    % Read email
    email = char(emails{row, 1});
    email = erase(email, 'Subject: ');
    
    % Change text to indice matrix
    word_indices = processEmail(email);
    X(row, :) = emailFeatures(word_indices);
    
end

emailsTable = table(X, y);
save('emailsTable.mat', 'emailsTable')

end