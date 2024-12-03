%% Load data
base_dir = "/Users/muhammadmahajna/workspace/research/code/cvr/src/model/output/";
sub_id = "SF_01135";

fmri_file = base_dir + sub_id + "_4D.mat";
cvr_file = base_dir + sub_id + "_CVR.mat";

load(fmri_file); % fmri_4d
load(cvr_file); % cvr_3d

%%

cvr_3d(cvr_3d<-0.7) = -0.7;
cvr_3d(cvr_3d>0.7) = 0.7;
fmri_4d(fmri_4d < 1) = nan;
figure;
imagesc(cvr_3d(:,end:-1:1,15)')
%%
figure;
imagesc(fmri_4d(:,end:-1:1,15, 10)')

sig1 = squeeze(fmri_4d(14,24,15, :));
sig2 = squeeze(fmri_4d(29,47,15, :));
sig3 = squeeze(fmri_4d(48,45,15, :));
sig4 = squeeze(fmri_4d(19,35,15, :));

plot(sig4, 'LineWidth',1)

%% CVR data visualization - spatial

figure;

ij = 1;
for slice=1:1:size(cvr_3d, 3)-1
    subplot(5,5,ij);
    data = cvr_3d(:,:, slice);
    imagesc(data)
    title(num2str(slice));
    ij=ij+1;
end

%% fMRI data visualization - spatial

timepoint = 10;
figure;

ij=1;
for slice=1:1:size(fmri_4d, 3)-1
    subplot(5,5,ij);
    data = fmri_4d(:,:, slice, timepoint);
    imagesc(data)
    title(num2str(slice) + " - " + num2str(timepoint));
    ij=ij+1;
end

%% CVR + FMRI - spatial
slice = 14;
figure;
ttr=3;
for i=1:ttr
    subplot(ttr,2,2*(i-1)+1);
    imagesc(cvr_3d(:,:,slice+i));

    subplot(ttr,2,2*i);
    imagesc(fmri_4d(:,:,slice+i,10));
end
%%
slice = 15;

ref_data = cvr_3d(:,:,slice);
fmri_data = squeeze(fmri_4d(:,:,slice,:));
fmri_data = 1*((fmri_data-min(fmri_data(:)))/(max(fmri_data(:))-min(fmri_data(:))))-0.5;
tt=all(fmri_data==0, 3);
%fmri_data = (fmri_data - mean(fmri_data,3))./std(fmri_data,[],3);

fmri_avg = mean(fmri_data, 3);
fmri_std = std(fmri_data, [],3);
figure;
subplot(1,3,1);
imagesc(ref_data);
subplot(1,3,2);
imagesc(fmri_avg);
subplot(1,3,3);
imagesc(fmri_std);

figure;
for i=20:25
    for j=20:25
        if(tt(i,j))
            fmri_data(i,j,:) = 0;
        end
        plot([squeeze(fmri_data(i,j,:))])
        hold on;
    end
end


%%

dt = zeros(64*64*26, 430);
ijk=1;
for i=1:64
    for j=1:64
        for k=1:26
            dt(ijk, :) = squeeze(fmri_4d(i,j,k,:));
            ijk=ijk+1;
        end
    end
end