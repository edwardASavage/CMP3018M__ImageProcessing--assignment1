% Task 5: Robust method --------------------------
clear; close all;

in_paths = ["IMG_01.jpg","IMG_02.jpg","IMG_03.jpg", ...
    "IMG_04.jpg","IMG_05.jpg","IMG_06.jpg","IMG_07.jpg", ...
    "IMG_08.jpg","IMG_09.jpg","IMG_10.jpg"];

GT_paths = ["IMG_01_GT.png","IMG_02_GT.png","IMG_03_GT.png", ...
    "IMG_04_GT.png","IMG_05_GT.png","IMG_06_GT.png","IMG_07_GT.png", ...
    "IMG_08_GT.png","IMG_09_GT.png","IMG_10_GT.png",];

% Perform enhancement, segmentation, and recognition on entire dataset.
% May take >30s to perform, consider reducing both in_paths and GT_paths
% to improve runtime.
segmentedImages = mySegment(in_paths);
labeledImages = myMultiLabel(segmentedImages);

for i = 1:length(labeledImages)
    figure = imshow(label2rgb(labeledImages{i},'prism','k','shuffle'));
% Uncomment line 21 for saving all processed labels to output file.
    % saveas(figure, [pwd '/output/img', num2str(i),'.fig']);
end




evalCells = myMultiEval(labeledImages, GT_paths);
T = cell2table(evalCells);
T.Properties.VariableNames = {'Dice score','Recall','Precision'};
T.Properties.RowNames = {'IMG_01','IMG_02','IMG_03', 'IMG_04', ...
    'IMG_05','IMG_06','IMG_07','IMG_08','IMG_09','IMG_10'};
% View table in workspace. this will show as cell for 'dice' column,
% view matlab variable to see full set.
T

% Pure function to return segmented images in type cell for parsed paths.
function segmentedImages = mySegment(in_paths)
% Initialize container for storing all segmented images.
segmentedImages = cell(length(in_paths),1);
    for i = 1:length(in_paths)
        Iorigin = imread(in_paths(i));
        I = rgb2gray(Iorigin);
        %Bicubic interpolation 
        I = imresize(I, 0.5);
        % Flatten with sigma 10, then contrast adjust.
        I = imflatfield(I,10);
        I = localcontrast(I, 0.5,1);
        % Use laplacian filter to edge aware blur.
        I = locallapfilt(I,0.4,3);
        % Open by reconstruction with square, adjust contrast.
        marker = imerode(I, strel('square',10));
        I = imreconstruct(marker, I);
        % Open by reconstruction with disk, then adjust contrast.
        I2 = imerode(I, strel('disk',20));
        I2 = imreconstruct(I2,I);
        I2 = localcontrast(I2, 0.1,1);
        % Binarize result.
        bw = imbinarize(I2);
        % Erode square, fill holes, then erode disk.
        bw = imerode(bw, strel('square',5));
        bw = imfill(~bw, 'holes');
        bw = imerode(bw, strel('disk',5));
        % Remove objects <5px.
        bw = bwareaopen(bw, 5);
        %Selected relevant cell entity and replace with result.
        segmentedImages{i} = bw;
    end
end

% Helper function for mass labeling of parsed segmented images.
function labeledImages = myMultiLabel(segs)
% Initialize cell container for labeled images.
labeledImages = cell(length(segs),1);
% Itterate through segmented images.
    for i = 1:length(segs)
        % Perform labeling
        labeledImages{i} = myLabelSeg(segs{i});
    end
end

% Helper function for mass evaluation stats of parsed image and truth
% pairs.
function multiEval = myMultiEval(Lmat, GT_paths)
% Initialize cell container for the three stats.
dice = cell(length(GT_paths),1);
recall =  cell(length(GT_paths),1);
precision = cell(length(GT_paths),1);
% Itterate through each pair (defined by length of ground truth).
    for i = 1:length(GT_paths)
        % Get Ground truth as labeled image.
        GT = double(imread(GT_paths(i)));
        % Perform evaluation, gather returns.
        [d, r, p] = myEvalSeg(Lmat{i}, GT);
        % Transpose Dice for row wise interpretation.
        dice{i} = d';
        recall{i} = r;
        precision{i} = p;
    end
    % Assign returned values to final multi cell.
    multiEval = [dice, recall, precision];
end

% Driver function for converting segmented images into respective labels.
function L_seg = myLabelSeg(bw)
% Gather  segmented labels properties
[objects, n] = bwlabel(bw);
stats = regionprops('table',objects,'Centroid',...
    'MajorAxisLength','MinorAxisLength');
% Init output matrix
L_seg = zeros(size(bw));
% For each element, find axis dimensions and compare them against defined
% paramaters for classification.
    for i = 1:n
        major = stats.MajorAxisLength(i);
        minor = stats.MinorAxisLength(i);
% Find exact row and column current object location.
        [r, c] = find(objects == i);
 % if difference is above 100, its a long screw,  above 15,its a screw. 
 % otherwise its a washer.
        if(major - minor > 90)
             L_seg(bwselect(bw, c, r)) = 3;
        elseif (major - minor > 20)
            L_seg(bwselect(bw, c, r)) = 2;
        else
            L_seg(bwselect(bw, c, r)) = 1;
        end
    end
end

% Driver function for evaluation stats.
function [d, r, p] =  myEvalSeg(A,B)
% Convert labeled images to type double
A = im2double(A);
B = im2double(B);
% return dice score
d = dice(A,B);
% return recall and precision statistics
r = sum(sum(A&B))/sum(B(:));
p = sum(sum(A&B))/sum(A(:));
end
