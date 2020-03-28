I=imread('Snr_Binning_1.tif');

I_histeq = histeq(I);
I_histadj=imadjust(I);
figure;


axis on

%hold off
imshow(I_histadj)
[xi,yi] = getpts

x1=[]
y1=[]
for i=1:length(xi)
    x1=[x1 round(xi(i))]
    y1=[y1 round(yi(i))]
end
[xoff yoff]=getpts

x2=x1 
y2=y1 - round(max(y1(i))-yoff(1))
I = double(I);

BW1 = roipoly(I_histadj,x1,y1);
BW1 = double(BW1);

BW1(BW1==0) = NaN;
filter_I1 = I.*BW1;
BW2 = roipoly(I_histadj,x2,y2);
BW2 = double(BW2);

BW2(BW2==0) = NaN;
filter_I2 = I.*BW2;

figure
axis on
subplot(2,2,1),imshow(uint8(BW1))
subplot(2,2,2),imshow(uint8(BW2))
subplot(2,2,3),imshow(uint8(filter_I1))
subplot(2,2,4),imshow(uint8(filter_I2))

mean_value1 = mean(filter_I1(~isnan(filter_I1)))
mean_value2 = mean(filter_I2(~isnan(filter_I2)))


