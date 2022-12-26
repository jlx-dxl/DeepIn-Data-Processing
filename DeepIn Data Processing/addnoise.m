clear all
clean = imread("320Hz_1\clean_2200.png");

%%
tic

I = clean;
N = length(I(:));
I = double(I);
P = I/sum(I(:));
Q = 10;
photonNum = Q*N;
% b = randsrc(photonNum, 1, [1:N;P(:)']);
% Iphoton = hist(b, N);
% Iphoton = reshape(Iphoton, size(I));
imshow(Iphoton)

toc
%%
I = double(clean);
Q = 1;
k = 65535 / Q;
poisson1 = imnoise(I / k, 'poisson');
poisson2 = poisson1 * k;
poisson = im2uint16(poisson2);
% poisson = typecast(poisson2(:),'uint16');
% poisson = reshape(poisson, size(I));
imshow(poisson)