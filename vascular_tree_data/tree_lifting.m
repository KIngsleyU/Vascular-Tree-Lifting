close all;
Sx=101;     Sy=101;     Sz=101;  %size of 3D volume 

for grp=1:10
    for im = 1:10
        %% reading the data
        f=fopen(['March_2013_VascuSynth_Dataset/Group',...
            num2str(grp),'/data',num2str(im),'/testVascuSynth',num2str(im),'_101_101_101_uchar.raw'],'r');
        %this data is downloaded from http://vascusynth.cs.sfu.ca/Data.html        
        I=fread(f,Sx*Sy*Sz,'uchar');
        fclose(f);
        
        %% saving training x and ground truth
        I=reshape(I,[Sx Sy Sz]);
        x=squeeze(max(I,[],1)); 
        for i = 1:4
            I = rot90(I);
            exportImage('training_data', I, grp, im + (i-1) * 12)
        end
    end
    
    for im = 11:12
        %% reading the data
        f=fopen(['March_2013_VascuSynth_Dataset/Group',...
            num2str(grp),'/data',num2str(im),'/testVascuSynth',num2str(im),'_101_101_101_uchar.raw'],'r');
        %this data is downloaded from http://vascusynth.cs.sfu.ca/Data.html        
        I=fread(f,Sx*Sy*Sz,'uchar');
        fclose(f);
        
        %% saving test x and ground truth
        I=reshape(I,[Sx Sy Sz]);
        x=squeeze(max(I,[],1)); 
        for i = 1:4
            I = rot90(I);
            exportImage('test_data', I, grp, im + (i-1) * 12)
        end
    end
end

%% saving train data
function exportImage(folder, I, grp, n)
    x=squeeze(max(I,[],1)); 
    groundTruthDir = append(folder, '/ground_truth/');
    inputDir =  append(folder, '/input/');
    name = [groundTruthDir,...
        num2str(grp),'_',num2str(n),'ground_truth.mat'];
    save(name,'I');
    imwrite(mat2gray(x),[inputDir,...
        num2str(grp),'_',num2str(n),'x.png']);
end

        %% simple rendering and depth calculation (Not Used)
%         I=reshape(I,[Sx Sy Sz]);
%         x=squeeze(max(I,[],1)); %simple approach, view direction along x 
%         depth=zeros(Sy,Sz);     %can rotate the volume prior to this step 
%         for y=1:Sy              %for other directions 
%             for z=1:Sz
%                 a=find(I(:,y,z));
%                 if ~isempty(a)
%                     depth(y,z)=a(1);
%                 end
%             end
%             
%         end
