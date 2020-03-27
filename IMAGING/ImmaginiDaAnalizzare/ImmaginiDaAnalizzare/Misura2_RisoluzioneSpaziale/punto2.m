
name='Binning_8.tif'
I=imread(name)
figure
imshow(I)

[rows columns depth]=size(I)

[xi,yi] = getpts
axis on
hold on;


plot(xi(1),yi(1), 'r+', 'MarkerSize', 30, 'LineWidth', 2);
plot(xi(2),yi(2), 'b+', 'MarkerSize', 30, 'LineWidth', 2);
hold off

a=round(xi(2))
b=round(xi(1))
gray=[]
distance=[]

for i=a:b
    intensity=I(round(yi(2)),i);
    gray=[gray,intensity]
    distance=[distance,i-a]
end
figure
grid on
hold on
plot(distance,gray),title(name),xlabel('distance [pixels]'),ylabel('gray intensity')
hold off
figname= strrep(name,'.tif','a')
saveas(gcf,sprintf('greay%s.png',figname))



