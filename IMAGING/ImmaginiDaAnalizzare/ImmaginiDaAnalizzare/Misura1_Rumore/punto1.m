
files = dir('Temperature65*.tif');

for i=1:length(files)
    name=files(i).name
    I = imread(name);
    %imtool(I)
    [M,N]=size(I);
    %i=rgb2gray(I)
    figure
    subplot(1,2,1), imshow(I),title(name)
    figname= strrep(name,'.tif','.png')
    subplot(1,2,2), imhist(I,500000),title('image gray-scale Histogram'),xlim([490 550]),ylim([0 1048576/4])
    saveas(figure,figname)
end

files = dir('Temperature25*.tif');

for i=1:length(files)
    name=files(i).name
    I = imread(name);
    %imtool(I)
    [M,N]=size(I);
    %i=rgb2gray(I)
    figure
    subplot(1,2,1), imshow(I),title(name)
    figname=strrep(name,'.tif','.png')
    subplot(1,2,2), imhist(I,500000),title('image gray-scale Histogram'),xlim([490 600]),ylim([0 1048576/4])
    saveas(figure,figname)
end