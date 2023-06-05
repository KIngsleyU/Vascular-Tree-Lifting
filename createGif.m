outputPath = fullfile(pwd, 'past results/todo');
progressPath = fullfile(outputPath, '*.png');
directory = dir(progressPath);
h = figure;
axis tight manual;
filename = 'output2.gif';
for i = 1:length(directory)
%     imageName = directory(i).name;
    imageName =  string(i) + '.png';
    imagePath = fullfile(outputPath, imageName);
    A = imread(imagePath);
    imagesc(A);
    drawnow
    frame = getframe(h);
    im = frame2im(frame);
    [image,map] = rgb2ind(im,256);
    if i == 1
       imwrite(image,map,filename,'gif','LoopCount',Inf,'DelayTime',0.25);
    else
       imwrite(image,map,filename,'gif','WriteMode','append','DelayTime',0.25);
    end
end
