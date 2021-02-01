clear; close all;
% Task 1: Pre-processing -----------------------
% Step-1: Load input image
I = imread('IMG_01.jpg');
% Step-2: Covert image to grayscale
I_gray = rgb2gray(I);
figure, imshow(I_gray), title('Grayscale image');
% Step-3: Rescale image
Ir = imresize(I_gray,0.5,'bilinear');
figure, imshow(Ir), title('0.5 bilinear interpoliation result');
% Step-4: Produce histogram before enhancing
figure,imhist(Ir), title('Histogram before enhancement');

% Step-5: Enhance image before binarisation
% Flatten with sigma = 2 for shadow flattening
Ie = imflatfield(Ir,2);
% Increase object clarity via edge aware contrast enhancement
Ie = localcontrast(Ie,1,1);
% Open image by reconstruction to reduce background elements
se = strel('disk',20);
Iobr = imerode(Ir,se);
Iobr = imreconstruct(Iobr,Ie);
% Readjust contrast using otsu method to enhance intestity at foreground
Iobr = imadjust(Iobr);
% Close opened image by reconstruction, 10x smaller strel for edge aware
% object intensity dilation.
se = strel('disk',2);
Icbr = imdilate(Iobr,se);
Icbr = imreconstruct(imcomplement(Icbr),imcomplement(Iobr));
Icbr = imcomplement(Icbr);
% Erode with square strel to complete remaining objects
Icbr = imerode(Icbr, strel('square',3));
% Readjust contrast using otsu method
Ie = imadjust(Icbr);
%figure,imshow(Ie);
figure,imshow(Ie), title('Enhanced image result');
% Step-6: Histogram after enhancement
figure,imhist(Ie), title('Histogram after enhancement');

% Step-7: Image Binarisation
bw = imbinarize(Ie);
bw = imfill(imcomplement(bw),'holes');
% figure,imshow(bw), title('object fill and binarization result');

% Task 2: Edge detection ------------------------
I_edge = edge(bw, 'canny');
figure,imshow(I_edge), title('Canny method edge detection');

% Task 3: Simple segmentation --------------------
L_seg = myLabelSeg(bw);
figure,imshow(L_seg), title('Segmented image result');

% Task 4: Object Recognition --------------------
L_seg = label2rgb(L_seg, 'prism','k','shuffle');
figure,imshow(L_seg), title('Automatic object recognition label result');

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
% if difference is above 20, its a screw. otherwise its a washer.
        if (major - minor > 20)
            L_seg(bwselect(bw, c, r)) = 2;
        else
            L_seg(bwselect(bw, c, r)) = 1;
        end
    end
end