function data = load_cropped_data(vidFrames, filter, scale)
    numFrames = size(vidFrames, 4);
    data = zeros(numFrames, 2);
    for j = 1:numFrames
        X = vidFrames(:,:,:,j);
        Xg = double(rgb2gray(X));
        X_cropped = Xg.* filter;
        threshold = X_cropped > scale;
        [Y, X] = find(threshold);
        data(j,1) = mean(X);
        data(j,2) = mean(Y);
    end
end
