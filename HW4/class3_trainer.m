function [U,S,V,w, vdigit1, vdigit2, vdigit3] = class3_trainer(digit1,digit2,digit3,feature)
    n1 = size(digit1,2);
    n2 = size(digit2,2);
    n3 = size(digit3,2);
    [U,S,V] = svd([digit1 digit2 digit3],'econ');
    genres = S*V'; % projection onto principal components
    U = U(:,1:feature);
    digit11 = genres(1:feature,1:n1);
    digit22 = genres(1:feature,n1+1:n1+n2);
    digit33 = genres(1:feature,n1+n2+1:n1+n2+n3);
    
    mdigit1 = mean(digit11,2);
    mdigit2 = mean(digit22,2);
    mdigit3 = mean(digit33,2);
    mtotal = mean([mdigit1 mdigit2 mdigit3],2);
    
    Sw = 0; % within class variances
    for k = 1:n1
        Sw = Sw + (digit11(:,k)-mdigit1)*(digit11(:,k)-mdigit1)';
    end
    for k = 1:n2
        Sw = Sw + (digit22(:,k)-mdigit2)*(digit22(:,k)-mdigit2)';
    end
    for k = 1:n3
        Sw = Sw + (digit33(:,k)-mdigit3)*(digit33(:,k)-mdigit3)';
    end
    Sb = size(digit1,2)*(mdigit1-mtotal)*(mdigit1-mtotal)' +...
        size(digit2,2)*(mdigit2-mtotal)*(mdigit2-mtotal)'...
        + size(digit3,2)*(mdigit3-mtotal)*(mdigit3-mtotal)';

    [V2,D] = eig(Sb,Sw); % linear discriminant analysis
    [~,ind] = max(abs(diag(D)));
    w = V2(:,ind); w = w/norm(w,2);
    
    vdigit1 = w'*digit11; 
    vdigit2 = w'*digit22;
    vdigit3 = w'*digit33;
    
end