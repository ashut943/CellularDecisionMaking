% Sizes and N
N = 1;
lmbda = 1;
n_I = 4 * (N + 1)^2;
n_p = 5 * N + 1;
n_s = n_I - 12;
n_P = 5 * N - 1;
n_t = n_p + n_s + 1;

disp('======================');
disp('Sizes and cost');
disp(['N: ', num2str(N)]);
disp(['lambda: ', num2str(lmbda)]);
disp(['nI: ', num2str(n_I)]);
disp(['np: ', num2str(n_p)]);
disp(['ns: ', num2str(n_s)]);
disp(['nP: ', num2str(n_P)]);
disp(['nt: ', num2str(n_t)]);

disp('======================');

% ~~~~~~~~~States
I = containers.Map('KeyType', 'double', 'ValueType', 'any');
Ikeyer = containers.Map('KeyType', 'char', 'ValueType', 'any');
T = [];
TG = [];
TB = [];
index = 0;

for ua = 0:N
    for ub = 0:N
        for sa = 0:1
            for sb = 0:1
                index = (2 * sa + sb) * (N + 1)^2 + (N + 1) * ua + ub + 1; % Adjusted for 1-indexing
                I(index) = [ua, ub, sa, sb];
                key = mat2str([ua, ub, sa, sb]);
                
                Ikeyer(key) = index;
            end
        end
    end
end

Ikeys = cell2mat(keys(I));
Ivals = values(I);

disp('State Definitions');
disp('All states');
for key = Ikeys
    value = I(key);
    disp([num2str(key), ': ', mat2str(value)]);
end

disp('~~~~~~~~~~~~~~~~~~~~~');

% Defining terminal states and TG TB
for ua = 0:N
    for ub = 0:N
        for sa = 0:1
            for sb = 0:1
                i = (2 * sa + sb) * (N + 1)^2 + (N + 1) * ua + ub + 1; % Adjusted for 1-indexing
                if ua == N && ub == 0
                    TG = [TG, i];
                    T = [T, i];
                end
                if ua == 0 && ub == N
                    TG = [TG, i];
                    T = [T, i];
                end
                if ua == N && ub == N
                    TB = [TB, i];
                    T = [T, i];
                end
            end
        end
    end
end
T = sort(T);
TG = sort(T);
TB = sort(T);

disp('Terminal States');
disp(['T: ', mat2str(T)]);
disp(['TG: ', mat2str(TG)]);
disp(['TB: ', mat2str(TB)]);

disp('======================');

% ~~~~~~~~~Parameters
f_plus = zeros(N, 2);
f_min = zeros(N, 2);
for i = 1:N
    for j = 1:2
        f_plus(i, j) = 1 + N * (j - 1) + (i - 1);
        f_min(i, j) = N * (j - 1) + (i - 1) + (2 * N) + 1;
    end
end

f_plus_flattened = f_plus(:)';
f_min_flattened = f_min(:)';
g = (4 * N + 1) + (0:N-1);
koff = 5 * N + 1;
P = [f_plus_flattened, f_min_flattened, g, koff];

disp('Parameters');
disp(['f+: ', mat2str(f_plus)]);
disp(['f-: ', mat2str(f_min)]);
disp(['g: ', mat2str(g)]);
disp(['k_off: ', num2str(koff)]);
disp(['P: ', mat2str(P)]);

disp('======================');

% ~~~~~~~~~Rate Matrix
M = zeros(n_I, n_I, n_p);
alpha = zeros(1, n_I);

for u1 = Ivals
    for u2 = Ivals
        % Extract contents from the cell arrays u and u_
        u = u1{1};
        u_ = u2{1};
        i = Ikeyer(mat2str(u));
        j = Ikeyer(mat2str(u_));
        % fprintf('u_ = [%d, %d, %d, %d], u = [%d, %d, %d, %d]\n', u, u_);
        if (isequal(u,u_))
            continue
        end
        ua = u(1); ub = u(2); sa = u(3); sb = u(4);
        ua_ = u_(1); ub_ = u_(2); sa_ = u_(3); sb_ = u_(4);

        flag = 0;
        tempk = 0;
        if j == i + (N + 1) && ub == ub_ && sa == sa_ && sb == sb_
            flag = 1;
            tempk = N * sa + ua+1;
            M(i, j, tempk) = flag;
        elseif j == i - (N + 1) && ub == ub_ && sa == sa_ && sb == sb_
            flag = 1;
            tempk = sa * N + ua + 2 * N ;
            M(i, j, tempk) = flag;
        elseif j == i + 1 && ua == ua_ && sa == sa_ && sb == sb_
            flag = 1;
            tempk = N * sb + ub+1;
            M(i, j, tempk) = flag;
        elseif j == i - 1 && ua == ua_ && sa == sa_ && sb == sb_
            flag = 1;
            tempk = sb * N + ub + 2 * N ;
            M(i, j, tempk) = flag;
        elseif j == i + 2 * (N + 1)^2 && ua == ua_ && ub == ub_ && ub ~= 0 && sb == sb_
            flag = 1;
            tempk = 4 * N + ub ;
            M(i, j, tempk) = flag;
        elseif j == i - 2 * (N + 1)^2 && ua == ua_ && ub == ub_ && sb == sb_
            flag = 1; 
            tempk = 5 * N+1;
            M(i, j, tempk) = flag;
        elseif j == i + (N + 1)^2 && ua == ua_ && ub == ub_ && ua ~= 0 && sa == sa_
            flag = 1;
            tempk = 4 * N + ua;
            M(i, j, tempk) = flag;
        elseif j == i - (N + 1)^2 && ua == ua_ && ub == ub_ && sa == sa_
            flag = 1;   
            tempk = 5 * N+1;
            M(i, j, tempk) = flag;
        end
    end
end
% Loop over state pairs
% for u1 = Ivals'
%     for u2 = Ivals'
%         % Extract contents from the cell arrays u and u_
%         u = u1{1};
%         u_ = u2{1};
%         fprintf('u_ = [%d, %d, %d, %d], u = [%d, %d, %d, %d]\n', u, u_);
%         i = Ikeyer(mat2str(u));
%         j = Ikeyer(mat2str(u_));
%         if (isequal(u, u_))
%             fprintf('i = %d, j = %d\n', i,j)
%             continue
%         end
%         ua = u(1); ub = u(2); sa = u(3); sb = u(4);
%         ua_ = u_(1); ub_ = u_(2); sa_ = u_(3); sb_ = u_(4);
% 
%         flag = 0;
%         tempk = 0;
% 
%         % Define transitions based on conditions
%         if j == i + (N + 1) && ub == ub_ && sa == sa_ && sb == sb_
%             flag = 1;
%             tempk = N * sa + ua;
%         elseif j == i - (N + 1) && ub == ub_ && sa == sa_ && sb == sb_
%             flag = 1;
%             tempk = sa * N + ua + 2 * N - 1;
%         elseif j == i + 1 && ua == ua_ && sa == sa_ && sb == sb_
%             flag = 1;
%             tempk = N * sb + ub;
%         elseif j == i - 1 && ua == ua_ && sa == sa_ && sb == sb_
%             flag = 1;
%             tempk = sb * N + ub + 2 * N - 1;
%         elseif j == i + 2 * (N + 1)^2 && ua == ua_ && ub == ub_ && ub ~= 0 && sb == sb_
%             flag = 1;
%             tempk = 4 * N + ub - 1;
%         elseif j == i - 2 * (N + 1)^2 && ua == ua_ && ub == ub_ && sb == sb_
%             flag = 1; 
%             tempk = 5 * N;
%         elseif j == i + (N + 1)^2 && ua == ua_ && ub == ub_ && ua ~= 0 && sa == sa_
%             flag = 1;
%             tempk = 4 * N + ua - 1;
%         elseif j == i - (N + 1)^2 && ua == ua_ && ub == ub_ && sa == sa_
%             flag = 1;   
%             tempk = 5 * N;
%         end
% 
%         if flag
%             M(i, j, tempk) = flag;
%         end
%     end
% end

disp('skipped')
% Sum across M to create diagonal elements
for k = 1:n_p
    M_i = sum(M(:, :, k), 2);
    for u = Ivals'
        i = Ikeyer(mat2str(u{1}));
        M(i, i, k) = -M_i(i);
    end
end

% Convert M to MATLAB array
Q__ = 0;
P_ = reshape(P, n_p, 1);
Q__ = zeros(n_I, n_I);
for k = 1:n_p
    Q__ = Q__ + P_(k) * M(:, :, k);
end
Q__ = squeeze(Q__);
disp(Q__)

% Define further matrices for the general non-linear problem
R = zeros(n_s, n_I);
Tc = setdiff(1:n_I, T);
i = 1;
for j = Tc
    R(i, j) = 1;
    i = i + 1;
end

Qtilde = R * Q__ * R';
Mtilde = einsum('ia,jb,abk->ijk', R, R, M);

% The rest of the code should follow the same structure as in Python
% Further matrices for the optimization problem, symmetric forms, etc.
