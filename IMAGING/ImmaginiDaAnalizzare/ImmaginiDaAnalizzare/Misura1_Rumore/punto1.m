
files = dir('Temperature65*.tif');
means1=[]
stds1=[]

for i=1:length(files)
    name=files(i).name
    I = imread(name);
    %imtool(I)
    [M,N]=size(I);
    %i=rgb2gray(I)
    figure
    subplot(1,2,1), imshow(I),title(name)
    figname= strrep(name,'.tif','')
    subplot(1,2,2), imhist(I,500000),title('image gray-scale Histogram'),xlim([490 550]),ylim([0 1048576/4])
    saveas(gcf,sprintf('HIS%d.png',figname))
    mean=mean2(I)
    std_deviation=std2(I)
    means1 = [means1 mean];
    stds1=[stds1 std_deviation];
end

files = dir('Temperature25*.tif');
means2=[]
stds2=[]
for i=1:length(files)
    name=files(i).name
    I = imread(name);
    %imtool(I)
    [M,N]=size(I);
    %i=rgb2gray(I)
    figure
    subplot(1,2,1), imshow(I),title(name)
    figname=strrep(name,'.tif','')
    subplot(1,2,2), imhist(I,500000),title('image gray-scale Histogram'),xlim([490 600]),ylim([0 1048576/4])
    saveas(gcf,sprintf('HIS%s.png',figname))
    mean=mean2(I)
    std_deviation=std2(I)
    means2 = [means2 mean];
    stds2=[stds2 std_deviation];
end
%disp(means1')
%disp(stds1)
%disp(means2')
%disp(stds2)
t1=[0.5; 1; 10; 60; 120; 180; 240; 300]
t2=[1; 10; 60; 120; 180; 240; 300]

f1=fit(t1,means1','poly1')

f1_2=fit(t1,stds1','poly1')

f2=fit(t2,means2','poly1')


f2_2=fit(t2,stds2','poly1')

figure
grid on
hold on
plot(f1,t1,means1'),title(' Mean vs Time'),xlabel('time [s]'),ylabel('mean [u.a]');
plot(f2,t2,means2');
hold off
legend('65 degree data','65 degree fit','25 degree data','25 degree fit')
saveas(gcf,sprintf('fittotal.png'))

figure
grid on
hold on
plot(f1_2,t1,stds1'),title(' Standard Dev vs Time'),xlabel('time [s]'),ylabel('std dev [u.a]');
plot(f2_2,t2,stds2');
hold off
legend('65 degree data','65 degree fit','25 degree data','25 degree fit')
saveas(gcf,sprintf('fittotal_stddev.png'))
%errorbar(t1,means1',stds1'),
%,errorbar(t2,means2',stds2')