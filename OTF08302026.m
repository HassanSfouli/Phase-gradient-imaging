addpath 'M:\MCXLAB\mcxlab_2025\mcxlab-win-x86_64-v2025 _ original\mcxlab';

addpath '\\nwz.wwu.de\dfs\Home\H\hsfouli\Desktop\outputs\iso2mesh-1.9.8-win64\iso2mesh';
addpath 'M:\MCXLAB\iso2mesh-1.9.8-win64\iso2mesh';
addpath 'M:\MCXLAB\mcxlab_2025\mcxlab-win-x86_64-v2025 _ original\mcxlab\matlab';
addpath 'G:\User\Hassan\Thesis\code per chapter\examples\Final_Folder';
addpath 'M:\MCXLAB\mcxlab_2025\mcxlab-win-x86_64-v2025 _ original\mcxlab';

n_oil = 1.455;
n_PMMA = 1.581;
n_quartz = 1.54;
n_air = 1.0;


nx = 128*2; ny = 128*2; nz = 50*5;


% Domain
cfg.vol = ones(nx,ny,nz);
cfg.vol(:,:,50:nz) = 2; 
cfg.vol(:,:,1:nz) = 2;

% MCX settings (fixed)

cfg.autopilot = 1;
cfg.gpuid     = 1;
cfg.issrcfrom0 = 1;
cfg.nphoton   = 5e8;  % Good balance speed/accuracy

cfg.srctype = 'ring';
cfg.srcpos  = [nx/2 ny/2 0];

% cfg.srctype = 'pencil';
% cfg.srcpos  = [nx/4 ny/2 -5];
cfg.angleinvcdf = linspace(0,1,100);
% cfg.angleinvcdf = linspace(0, 1 ,100);

% angles = [3.1416 4.7124]+3*3.1416/4;  % Use one illumination direction
angles = [3.1416 4.7124]+1*3.1416/4;  % Use one illumination direction
cfg.srcparam1 = [60*2 1*30*2 angles(1,1) angles(1,2)];


r = 1;
d = 20;
cfg.detpos = [64*2 64*2 0*nz d];
        
theta = 0;
% figure(), mcxpreview(cfg);title('domain preview');
cfg.srcdir = [sin(theta) 0 cos(theta)];

cfg.tstart = 0;
cfg.tend = 5e-9;
cfg.tstep = 5e-9;
cfg.issrcfrom0 = 1;

cfg.issaveexit = 1;
cfg.isreflect = 0;         %  [1]-consider refractive index mismatch, 0-matched index
cfg.bc = 'aa_aa_';
cfg.isnormalized = 1;      %[1]-normalize the output fluence to unitary source, 0-no reflection.
cfg.isspecular = 1;        % 1-calculate specular reflection if source is outside, [0] no specular reflection
cfg.unitinmm = 3.33e-5/2;        %defines the length unit for a grid edge length [1.0]
cfg.unitinmm  = 0.02;
% cfg.unitinmm  = 1;

cfg.gpuid = 1;
cfg.autopilot = 1;
cfg.issrcfrom0 = 1; 
 
cfg.maxdetphoton = 1e7;

% ==========================
% ANGULAR GRID (fixed)
% ==========================

grid_size = 128;  % Smaller for faster sweep


u_edges = linspace(-2,2,grid_size);
v_edges = linspace(-2,2,grid_size);
% u_edges = linspace(-1,1,grid_size);
% v_edges = linspace(-1,1,grid_size);
u_cent = (u_edges(1:end-1)+u_edges(2:end))/2;
v_cent = (v_edges(1:end-1)+v_edges(2:end))/2;

[U,V] = meshgrid(u_cent, v_cent);
du = u_cent(2)-u_cent(1);
dv = v_cent(2)-v_cent(1);

NA = 0.6;
pupil = (U.^2 + V.^2) < NA.^2;
obliquity = sqrt(max(0,1-U.^2-V.^2));
obliquity(isnan(obliquity)) = 0;


% Tissues optical properties
%     0 0   1   n_quartz;
% mua = 0.019;  mus_values = 7.8;   g_values   = 0.89;  n_WM  = 1.37;  % Scalp & skull
 % mua = 0.004;  mus_values = 0.009; g_values   = 0.89;  n_WM  = 1.37;  % CSF
% mua = 0.02;  mus_values = 9.0;    g_values   = 0.89;  n_WM  = 1.37;  % Gray-matter
mua = 0.08;  mus_values = 40.9;   g_values   = 0.84;  n_WM  = 1.37;  % White-matter


g = g_values;
mus = mus_values;


% Set optical properties
cfg.prop = [ ...
    0       0        1   1.0;
    0 0   1   n_quartz;
    mua     mus      g   n_WM];

% Run MCX
tic;
[fluence, detphoton] = mcxlab(cfg);
toc;

% Extract photon directions
ud = detphoton.v(:,1);
vd = detphoton.v(:,2);
wd = detphoton.v(:,3);


mask = wd < 0;
ud = ud(mask);
vd = vd(mask);

% Histogram
binx = discretize(ud, u_edges);
biny = discretize(vd, v_edges);

valid = ~isnan(binx) & ~isnan(biny);

lin_idx = sub2ind([length(v_cent) length(u_cent)], ...
                  biny(valid), binx(valid));

h = accumarray(lin_idx, 1, [length(v_cent)*length(u_cent), 1]);
h = reshape(h, length(v_cent), length(u_cent));

% Compute S(u)
S = h .* obliquity;
S = S ./ (sum(S(:)) * du * dv);

% Compute WOTF
A = S;
B = pupil;

% FFT-based cross-correlation
FA = fft2(ifftshift(A));
FB = fft2(ifftshift(B));

corr = fftshift(real(ifft2(conj(FA) .* FB)));

% WOTF for phase
C = corr;
T_phi = -1i * (C - conj(flipud(fliplr(C))));
T_phi = T_phi / max(abs(T_phi(:)));

% Calculate slope of T_phi at q=0
N = size(T_phi, 1);
% q = linspace(-2*NA, 2*NA, N);
q = linspace(-1, 1, N);
[~, mid] = min(abs(q));

% Get cross-section along qx
T_qx = imag(T_phi(:,mid));

% Fit near zero (use 5 points on each side)
fit_range = 5;
idx_range = mid-fit_range : mid+fit_range;

% Ensure indices are valid
if min(idx_range) >= 1 && max(idx_range) <= length(q)
    q_fit = q(idx_range);
    T_fit = T_qx(idx_range);
    
    % Linear fit
    p = polyfit(q_fit, T_fit, 1);
    slope_T = p(1);
else
    slope_T = NaN;
end

% Store results
mus_red = mus * (1 - g);


% Frequency axis q normalized to NA (Limit is 2*NA)
%% We calculate the axis based on the padded size
 % T_WM = T_phi;
 % T_GM = T_phi;
 T_SS = T_phi;
 % T_CSF = T_phi;

N_total = size(T_phi, 1);
q_max = 2.0 * (N_total / (2 * N)); % Adjusts for padding ratio
q_axis = linspace(-q_max, q_max, N_total);

figure('Name', 'Final DPC Transfer Function');
    imagesc(q_axis, q_axis, imag(T_phi));
    axis image; colorbar; colormap parula;
    % hold on;
    % % Draw a circle at q=1 (NA) and q=2 (2*NA) to verify support
    % phi_circle = linspace(0, 2*pi, 100);
    % plot(NA*cos(phi_circle), NA*sin(phi_circle), 'k--', 'LineWidth', 2);
    % plot(2*cos(phi_circle), 2*sin(phi_circle), 'k-', 'LineWidth', 2);
    title('Imag{T(q)}: Support extends to 2 \times NA (Outer Circle)');
    xlabel('Normalized Frequency (q/NA)');
    ylabel('Normalized Frequency (q/NA)');
    xlabel('u');
    ylabel('v');


%%
uCenters = (u_edges(1:end-1) + u_edges(2:end))/2;
vCenters = (v_edges(1:end-1) + v_edges(2:end))/2;

% Contour levels
levels = [-0.2 0.2];
figure

contour(uCenters, vCenters, imag(TWM), levels, ...
    'b', 'LineWidth', 2)
hold on

contour(uCenters, vCenters, imag(TGM), levels, ...
    'r--', 'LineWidth', 2)

contour(uCenters, vCenters, imag(TSS), levels, ...
    'g:', 'LineWidth', 2)

contour(uCenters, vCenters, imag(TCSF), levels, ...
    'k-.', 'LineWidth', 2)

axis equal tight
set(gca, 'YDir', 'reverse')

legend('WM','GM','SS','CSF')
title('Contour comparison')
xlabel('u')
ylabel('v')
grid on



%% TRANSFER FUNCTION COMPUTATION

grid_size = 128;
u_edges = linspace(-1, 1, grid_size);
v_edges = linspace(-1, 1, grid_size);
u_cent = (u_edges(1:end-1) + u_edges(2:end))/2;
v_cent = (v_edges(1:end-1) + v_edges(2:end))/2;
[U, V] = meshgrid(u_cent, v_cent);
du = u_cent(2) - u_cent(1);
dv = v_cent(2) - v_cent(1);

NA = 0.4;
pupil = (U.^2 + V.^2) < NA^2;
obliquity = sqrt(max(0, 1 - U.^2 - V.^2));
obliquity(isnan(obliquity)) = 0;

%%

ud = detphoton.v(:,1);
vd = detphoton.v(:,2);
wd = detphoton.v(:,3);

mask = wd < 0;
ud = ud(mask);
vd = vd(mask);

binx = discretize(ud, u_edges);
biny = discretize(vd, v_edges);
valid = ~isnan(binx) & ~isnan(biny);
lin_idx = sub2ind([length(v_cent), length(u_cent)], biny(valid), binx(valid));
h = accumarray(lin_idx, 1, [length(v_cent)*length(u_cent), 1]);
h = reshape(h, length(v_cent), length(u_cent));

S = h .* pupil .* obliquity;
S = S ./ (sum(S(:)) * du * dv);

% Compute T(q) from S(u)
q_max = 2 * NA;
Nq = 128;
q_freq = linspace(-q_max, q_max, Nq);
[Qx, Qy] = meshgrid(q_freq, q_freq);
dq = q_freq(2) - q_freq(1);
P = double(sqrt(U.^2 + V.^2) <= NA);
T_phi = zeros(Nq, Nq);

for i = 1:Nq
    for j = 1:Nq
        qx = Qx(i,j);
        qy = Qy(i,j);
        
        if abs(qx) < 1e-10 && abs(qy) < 1e-10
            T_phi(i,j) = 0;
            continue;
        end
        
        integral_sum = 0;
        
        for iu = 1:length(u_cent)
            for ju = 1:length(v_cent)
                u_prime = u_cent(iu);
                v_prime = v_cent(ju);
                
                u_plus = u_prime + qx/2;
                v_plus = v_prime + qy/2;
                u_minus = u_prime - qx/2;
                v_minus = v_prime - qy/2;
                
                if abs(u_plus) <= max(u_cent) && abs(v_plus) <= max(v_cent) && ...
                   abs(u_minus) <= max(u_cent) && abs(v_minus) <= max(v_cent)
                    
                    i_plus = round((u_plus - u_cent(1))/du) + 1;
                    j_plus = round((v_plus - v_cent(1))/dv) + 1;
                    i_minus = round((u_minus - u_cent(1))/du) + 1;
                    j_minus = round((v_minus - v_cent(1))/dv) + 1;
                    
                    if i_plus >= 1 && i_plus <= length(u_cent) && ...
                       j_plus >= 1 && j_plus <= length(v_cent) && ...
                       i_minus >= 1 && i_minus <= length(u_cent) && ...
                       j_minus >= 1 && j_minus <= length(v_cent)
                        
                        S_diff = S(i_plus, j_plus) - S(i_minus, j_minus);
                        P_product = P(i_plus, j_plus) * P(i_minus, j_minus);
                        integral_sum = integral_sum + S_diff * P_product * du * dv;
                    end
                end
            end
        end
        T_phi(i,j) = -1i * integral_sum;
    end
end





%% Angular illumination distribution on a unit hemisphere
% detphoton.v(:,1:3) contains photon direction vectors

% Direction cosines
uPhoton =  detphoton.v(:,1);
vPhoton =  detphoton.v(:,2);
wPhoton = -detphoton.v(:,3);   % Flip sign if photons travel toward -z
% Keep photons in the upper hemisphere
valid = isfinite(uPhoton) & isfinite(vPhoton) & isfinite(wPhoton) & ...
        (wPhoton > 0);

uPhoton = uPhoton(valid);
vPhoton = vPhoton(valid);
wPhoton = wPhoton(valid);

% Normalize vectors, in case MCX output is not perfectly normalized
directionNorm = sqrt(uPhoton.^2 + vPhoton.^2 + wPhoton.^2);

% uPhoton = uPhoton ./ directionNorm;
% vPhoton = vPhoton ./ directionNorm;
% wPhoton = wPhoton ./ directionNorm;

% Histogram in normalized spatial-frequency coordinates
nBins = 25;

uEdges = linspace(-1,1,nBins+1);
vEdges = linspace(-1,1,nBins+1);

photonCounts = histcounts2(uPhoton,vPhoton,uEdges,vEdges);

uCenters = (uEdges(1:end-1) + uEdges(2:end))/2;
vCenters = (vEdges(1:end-1) + vEdges(2:end))/2;

% histcounts2 returns rows corresponding to u and columns to v.
% Transpose so matrices agree with meshgrid coordinates.
photonCounts = photonCounts.';

[U,V] = meshgrid(uCenters,vCenters);

% Upper unit hemisphere
R2 = U.^2 + V.^2;
insideHemisphere = R2 <= 1;

W = zeros(size(U));
W(insideHemisphere) = sqrt(1-R2(insideHemisphere));

% Convert counts into an effective source distribution
du = uEdges(2)-uEdges(1);
dv = vEdges(2)-vEdges(1);

% Photon density per unit area in the u-v plane
sourceDistribution = photonCounts/(du*dv);

% % Optional obliquity correction.
% % Remove this line if only a raw projected histogram is required.
% minimumW = 0.05;
% sourceDistribution = sourceDistribution ./ max(W,minimumW);
% 
% sourceDistribution(~insideHemisphere) = 0;

% Smooth the Monte Carlo histogram
% imgaussfilt requires Image Processing Toolbox
if exist('imgaussfilt','file')
    sourceDistribution = imgaussfilt(sourceDistribution,2);
else
    % Toolbox-independent Gaussian smoothing
    sigma = 2;
    filterRadius = ceil(3*sigma);
    q = -filterRadius:filterRadius;
    gaussianKernel = exp(-(q.^2)/(2*sigma^2));
    gaussianKernel = gaussianKernel/sum(gaussianKernel);

    sourceDistribution = conv2(gaussianKernel,...
                               gaussianKernel,...
                               sourceDistribution,...
                               'same');
end

sourceDistribution(~insideHemisphere) = 0;

% Normalize for display
sourceDistribution = sourceDistribution ./ ...
                     max(sourceDistribution(:),[],'omitnan');

% Plot the base plane
figure('Color','w','Position',[100 100 720 600]);
        baseDistribution = sourceDistribution;
        baseDistribution(~insideHemisphere) = 0;
        % surfaceHandle = surf(U,V,zeros(size(U)),baseDistribution,...
        %     'EdgeColor','none',...
        %     'FaceColor','interp');
        % 
        % hold on;

        % Plot colored hemisphere
        surf(U,V,W,sourceDistribution,...
            'FaceColor','interp',...
            'EdgeColor','k',...
            'LineWidth',0.25);
        % Formatting
        axis equal;
        axis([-1 1 -1 1 0 1.05]);
        xlabel('u','FontSize',16);
        ylabel('v','FontSize',16);
        zlabel('w','FontSize',16);
        view(135,25);
        grid on;
        box on;
        colormap(gca, gray)
        colorbar;
        set(gca,...
            'FontSize',13,...
            'LineWidth',1,...
            'XTick',[-1 -0.5 0 0.5 1],...
            'YTick',[-1 -0.5 0 0.5 1],...
            'ZTick',[0 0.5 1]);
title('Angular Distribution of Illumination');



% Direction of maximum illumination intensity
% sourceDistribution must correspond to the colors plotted on U,V,W
S = sourceDistribution;
% S(~insideHemisphere) = NaN;

% Find maximum-intensity grid cell
[~, linearIndex] = max(S(:), [], 'omitnan');
[rowPeak, colPeak] = ind2sub(size(S), linearIndex);

uPeak = U(rowPeak, colPeak);
vPeak = V(rowPeak, colPeak);
wPeak = sqrt(max(0, 1 - uPeak^2 - vPeak^2));
% wPeak = W(rowPeak,colPeak);

% Normalize for safety
peakDirection = [uPeak, vPeak, wPeak];
peakDirection = peakDirection / norm(peakDirection);

uPeak = peakDirection(1);
vPeak = peakDirection(2);
wPeak = peakDirection(3);
            
        % Angles
        alphaPeakDeg = acosd(max(-1,min(1,uPeak)));
        betaPeakDeg  = acosd(max(-1,min(1,vPeak)));
        gammaPeakDeg = acosd(max(-1,min(1,wPeak)));
        
        % Optional spherical coordinates
        rhoPeak = hypot(uPeak,vPeak);
        phiPeakDeg = atan2d(vPeak,uPeak);  % azimuth in u-v plane
        thetaPeakDeg = gammaPeakDeg;       % polar angle from +w
        
        fprintf('Peak direction:\n');
        fprintf('u = %.4f, v = %.4f, w = %.4f\n', uPeak, vPeak, wPeak);
        fprintf('phi = %.2f deg\n', phiPeakDeg);
        fprintf('theta = %.2f deg\n', thetaPeakDeg);
        
        % Draw arrow toward maximum-intensity region
        hold on;
        % Dashed direction line + incoming arrow
        
        Ppeak = [uPeak, vPeak, wPeak];
        
        % Full dashed line along the photon direction
        % Starts near the origin and extends outside the hemisphere
        lineStart = [0 0 0];
        lineEnd   = 2.5 * Ppeak;
        
        plot3([lineStart(1), lineEnd(1)], ...
              [lineStart(2), lineEnd(2)], ...
              [lineStart(3), lineEnd(3)], ...
              '--k', ...
              'LineWidth', 1.8);

% Incoming arrow near the surface
% Arrow points from outside toward the peak
Pstart = 1.70 * Ppeak;
Pend   = 1.32 * Ppeak;
D      = Pend - Pstart;

        quiver3(Pstart(1), Pstart(2), Pstart(3), ...
                D(1), D(2), D(3), ...
                0, ...
                'Color', 'r', ...
                'LineWidth', 2.5, ...
                'MaxHeadSize', 0.8);
        % Mark the peak direction on the hemisphere
        plot3(Ppeak(1), Ppeak(2), Ppeak(3), ...
              'ko', ...
              'MarkerFaceColor', 'w', ...
              'MarkerSize', 8, ...
              'LineWidth', 1.5);
        % Label
        Ptext = 1.48*Ppeak;
        labelText = sprintf([...
            ['\\alpha = %.1f^\\circ,\n' ...
            ' \\beta = %.1f^\\circ\n'],...
            '\\gamma = %.1f^\\circ\n'],...
            alphaPeakDeg,betaPeakDeg,...
            gammaPeakDeg);
        text(Ptext(1),Ptext(2),Ptext(3),labelText,...
            'Interpreter','tex',...
            'FontSize',10,...
            'FontWeight','bold',...
            'HorizontalAlignment','center',...
            'VerticalAlignment','bottom',...
            'BackgroundColor','w',...
            'Margin',3,...
            'Clipping','off');
        % Formatting
        axis equal;
        axis([-1.5 1.5 -1.5 1.5 0 1.55]);
        xlabel('u','FontSize',16);
        ylabel('v','FontSize',16);
        zlabel('w','FontSize',16);
        title('Angular Distribution of Incoming Illumination');
        view(135,25); grid on; box on; colorbar;
        set(gca,...
            'FontSize',13,...
            'LineWidth',1,...
            'XTick',[-1 -0.5 0 0.5 1],...
            'YTick',[-1 -0.5 0 0.5 1],...
            'ZTick',[0 0.5 1]);


%%
[flux, detpt, vol, seeds] = mcxlab(cfg);
newcfg = cfg;
newcfg.seed = seeds.data;
newcfg.outputtype = 'jacobian';
newcfg.detphotons = detpt.data;
newcfg.maxjumpdebug = 29371108;

newcfg.traj = 1;       % enable trajectory saving
newcfg.trajtype = 0;   % 0 = save for all photons, not just detected ones
newcfg.outputtype = 'jacobian'; % ensure flux is saved too if needed
newcfg.savetrace = 1;  % or savetrajectory
[flux2, detpt2, vol2, seeds2, traj2] = mcxlab(newcfg);

mcxplotvol(log10(abs(flux2.data)))
Flux = log10(abs(flux2.data));

figure(), 
    imagesc(flipud(squeeze(Flux(30,:,:))'))
    colormap(gca,jet)

F =  imgaussfilt(flipud(squeeze(Flux(nx/2,:,:))'),1.5);


spatialax = linspace(1,nx,nz)*cfg.unitinmm;
spatialaz = linspace(1,nz,nz)*cfg.unitinmm;

figure()
    imagesc(spatialax, spatialaz, squeeze(Flux(nx/2,:,:))')
    colormap(gca,jet)
    axis image
    xlabel("y (mm)")
    ylabel("z (mm)")
    set(gca, 'YDir', 'normal')  % Ensures y increases upward (bottom = lowest value)

    figure()
    imagesc(spatialax, spatialaz, squeeze(Flux(:,nx/2,:))')
    colormap(gca,jet)
    axis image
    xlabel("y (mm)")
    ylabel("z (mm)")
    set(gca, 'YDir', 'normal')  % Ensures y increases upward (bottom = lowest value)


figure()
    imagesc(F)
    colormap(gca,jet)
    axis image
    hold on
    contour(F,20,'k','LineWidth',0.5)
    colormap(gca,"jet")
    colorbar