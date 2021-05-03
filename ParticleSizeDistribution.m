function radius=ParticleSizeDistribution(fileImage, edgeWidthRemove, edgeHigthRemove, Resolution, levelGray, numberOfBins)
%   Particle Size Distribution
%   Created by yoga hariman
%   Copyright 2019
%   https://github.com/yogahariman

% clear
% clc
% close all
% 
% edgeWidthRemove = 0.05; %0-1
% edgeHigthRemove = 0.05; %0-1
% Resolution=0.9243548387096774; % micron/pixel % this is the spatial resolution of the input 
% numberOfBins=30; % This is the number of bars for histogram chart
% 
% nameFile = {...
%             'frel high 1'...
%             'frel high 2'...
%             'frel high 3'...
%             'frel high 4'...
% %             'frel medium 1'...
% %             'frel medium 2'...
% %             'frel medium 3'...
% %             'frel medium 4'...
% };
% 
% radius = [];
% for ii=1:length(nameFile)
%     fileImage=['/Drive/E/BatuRio_202010/Image/' nameFile{ii} '.png'];
%     
%     imgOriginal = imread(fileImage);
%     [m, n, ~]=size(imgOriginal);
%     ma = ceil(m*edgeHigthRemove);
%     na = ceil(n*edgeWidthRemove);
%     imgOriginal = imgOriginal(ma+1:m-ma,na+1:n-na,:);
%     imgGray = rgb2gray(imgOriginal); imgGray = 255-imgGray;
%     levelGray = graythresh(imgGray);
%     
%     temp = ParticleSizeDistribution(fileImage, edgeWidthRemove, edgeHigthRemove, Resolution, levelGray, numberOfBins);
%     radius=[radius;temp'];
% end
% 
% fig3=figure();
% histogram(radius, numberOfBins);
% [radius_Frequencies,edges] = histcounts(radius,numberOfBins);
% xlabel('Equivalent Particle Radius (micron)');
% ylabel('Relative Frequency')
% 
% y_freq = radius_Frequencies;
% x_radius = edges(1:end-1)+(edges(2:end)-edges(1:end-1))./2;
% save('/Drive/E/BatuRio_202010/Image/frel medium all_Hist','y_freq','x_radius');
% saveas(fig3,'/Drive/E/BatuRio_202010/Image/frel medium all_Hist.jpg');


imgOriginal = imread(fileImage);
% figure();imshow(imgOriginal);
[filepath,name,~] = fileparts(fileImage);

% potong image
[m, n, ~]=size(imgOriginal);
ma = ceil(m*edgeHigthRemove);
na = ceil(n*edgeWidthRemove);

imgOriginal = imgOriginal(ma+1:m-ma,na+1:n-na,:);
% figure();imshow(imgOriginal);


% rgb2gray
imgGray = rgb2gray(imgOriginal); imgGray = 255-imgGray;
% figure();imshowpair(imgOriginal,imgGray,'montage');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% binary image
% levelGray = graythresh(imgGray);
A = imbinarize(imgGray, levelGray);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% CALCULATIONS
Conn=8;
[s1,s2]=size(A);
A=~bwmorph(A,'majority',10); % figure(); imshow(A);
% Poro=sum(sum(~A))/(s1*s2);
D=-bwdist(A,'chessboard');
B=medfilt2(D,[3 3]);
B=watershed(B,Conn);
Pr=zeros(s1,s2);

for I=1:s1
    for J=1:s2
        if A(I,J)==0 && B(I,J)~=0
            Pr(I,J)=1;
        end
    end
end

% Remove small objects from binary image
Pr=bwareaopen(Pr,9,Conn);

cc = bwconncomp(Pr,8);
% imgLabel = label2rgb(labelmatrix(cc));
% figure();imshow(label2rgb(imgLabel));

measurements = regionprops(cc, {'Area','centroid'});
jumlahPixel = [measurements.Area];

radius = Resolution.*(jumlahPixel./pi).^.5; % grain radius


fig1 = figure();
imshow(imgOriginal);
hold on
centroids = cat(1,measurements.Centroid);
plot(centroids(:,1),centroids(:,2),'co')
hold off

fig2 = figure();
histogram(radius, numberOfBins);
[radius_Frequencies,edges] = histcounts(radius,numberOfBins);
xlabel('Equivalent Particle Radius (micron)');
ylabel('Relative Frequency')


y_freq = radius_Frequencies';
x_radius = (edges(1:end-1)+(edges(2:end)-edges(1:end-1))./2)';
save([filepath,'/',name,'_xy.mat'],'y_freq','x_radius');
saveas(fig1,[filepath,'/',name,'_Center.jpeg']);
saveas(fig2,[filepath,'/',name,'_hist.jpeg']);

close(fig1);
close(fig2);
