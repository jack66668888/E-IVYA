function [BestSol_Cost, BestSol_Position, Convergence_curve] = E_IVYA(N, Max_iteration, lb, ub, dim, fobj)
% E-IVYA: A Synergistic Enhancement of the Ivy Algorithm for GAN-based Imbalanced Classification
% This MATLAB code implements the Enhanced Ivy Algorithm (E-IVYA) as described in the paper.
% It integrates three synergistic mechanisms into the original IVYA to improve performance.

% ----- MODIFIED BASED ON REVIEWER R6 FEEDBACK (Tasks 5 & 6) -----

%% Problem Definition
CostFunction = @(x) fobj(x);
VarSize = [1, dim]; % Decision Variables Matrix Size
VarMin = lb;       % Lower Bound of Variables
VarMax = ub;       % Upper Bound of Variables

%% E-IVYA Parameters
MaxIt = Max_iteration; % Maximum Number of Iterations
nPop = N;              % Population Size

% --- Parameters for Enhancement Strategies ---
% Strategy 1: Dynamic Perturbation Framework
tk_pert = 0.05; % Asymmetric perturbation probability (tk)
c1 = 2;         % Symmetric exploration exponent
c2 = 2;         % Asymmetric exploration exponent

% Strategy 2: Elite Differential Mutation Escape Mechanism
MaxStagnation = 30; % Stagnation threshold
p = 0.1;            % Elite archive percentage
F = 0.5;            % DE scaling factor

% Strategy 3: Adaptive Movement Strategy (SCA)
a = 2; % SCA initial control parameter

%% Initialization
% Empty Plant Structure
empty_plant.Position = [];
empty_plant.Cost = [];
empty_plant.GV = [];

pop = repmat(empty_plant, nPop, 1); % Initial Population Array
stagnation_count = 0;              % Initialize stagnation counter

for i = 1:nPop
    % Initialize Position (Eq. 1 in original paper)
    pop(i).Position = unifrnd(VarMin, VarMax, VarSize);
    
    % Initialize Growth Vector (GV)
    pop(i).GV = (pop(i).Position ./ (VarMax - VarMin));
    
    % Evaluation
    pop(i).Cost = CostFunction(pop(i).Position);
end

% Sort Population and find Best Solution
[~, SortOrder] = sort([pop.Cost]);
pop = pop(SortOrder);
BestSol = pop(1);

% Initialize Convergence Curve
Convergence_curve = zeros(MaxIt, 1);

%% E-IVYA Main Loop
for it = 1:MaxIt
    
    % ----------------------------------------------------------------------
    % PHASE A: CANDIDATE SOLUTION GENERATION (INNOVATION 3: ADAPTIVE MOVEMENT)
    % ----------------------------------------------------------------------
    
    newpop = repmat(empty_plant, nPop, 1);
    
    % Calculate dynamic SCA parameter r1 for the current iteration
    r1 = a - a * (it / MaxIt);
    
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % R6 FIX (Task 6): Start of modification
    % Define the elite group size (consistent with Phase C logic)
    % This is used to select a 'more vital neighbor' 
    n_elite_neighbor = max(3, ceil(nPop * p));
    n_elite_neighbor = min(n_elite_neighbor, nPop); % Cap at nPop
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    for i = 1:nPop
        
        % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % R6 FIX (Task 6): Clarify Movement Rule 
        % The original code used a simple 'next index' (ii = i+1), which
        % contradicts the paper's "more vital neighbor" concept.
        % We now implement 'ii' as a randomly selected 'more vital'
        % individual from the elite group (top n_elite_neighbor).
        
        ii = randi(n_elite_neighbor); % Select a random index from the sorted top
        
        % Ensure 'ii' is not 'i' to have a meaningful differential vector
        % in Eq. 16. This is especially important if 'i' is also in the elite group.
        if ii == i
            if n_elite_neighbor > 1 % If there are other elites to choose from
               % Find a different elite neighbor
               while ii == i
                   ii = randi(n_elite_neighbor);
               end
            else
               % Edge case: nPop=1 or n_elite_neighbor=1. Just use index 1.
               ii = 1;
            end
        end
        % --- End of R6 FIX (Task 6) Block ---
        % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        % Beta condition from original IVYA
        beta_1 = 1 + (rand / 2);

        % Generate random parameters for SCA
        r2 = 2 * pi * rand();
        r3 = 2 * pi * rand();
        r4 = rand();
        
        if pop(i).Cost < beta_1 * pop(1).Cost
            % Local Search with SCA update (Eq. 16 in the paper)
            % This now learns from a 'more vital neighbor' (ii)
            term1 = r1 .* sin(r2) .* (abs(randn(VarSize)) .* (pop(ii).Position - pop(i).Position));
            term2 = r1 .* cos(r3) .* (randn(VarSize) .* pop(i).GV);
            newpop(i).Position = pop(i).Position + term1 + term2;
        else
            % Global Search with SCA update (Eq. 17 in the paper)
            term1 = r1 .* sin(r2) .* (r4 .* pop(1).Position - pop(i).Position);
            term2 = r1 .* cos(r3) .* pop(i).GV;
            newpop(i).Position = pop(i).Position + term1 + term2;
        end
        
        % Update Growth Vector of the original individual (Eq. 3 in original paper)
        pop(i).GV = pop(i).GV .* ((rand^2) * randn(1, dim));

        % Boundary Checking
        newpop(i).Position = max(newpop(i).Position, VarMin);
        newpop(i).Position = min(newpop(i).Position, VarMax);
        
        % Update Growth Vector for the new position
        newpop(i).GV = (newpop(i).Position ./ (VarMax - VarMin));

        % Evaluate new solution
        newpop(i).Cost = CostFunction(newpop(i).Position);
    end
    
    % Merge, Sort, and Select
    pop = [pop; newpop];
    [~, SortOrder] = sort([pop.Cost]);
    pop = pop(SortOrder);
    pop = pop(1:nPop);

    % ----------------------------------------------------------------------
    % PHASE B: DIVERSITY ENHANCEMENT (INNOVATION 1: DYNAMIC PERTURBATION)
    % ----------------------------------------------------------------------
    
    % --- Symmetric Exploration Operator (Elite-Guided OBL) ---
    m_obl = (1 + it/MaxIt)^c1; % Eq. 9
    Center = (VarMin + VarMax) / 2;
    for k = 1:ceil(nPop/2) % Apply to a portion of the population
        % Generate opposite individual using elite guidance (Eq. 10)
        OppositePosition = Center + (Center - pop(1).Position) / m_obl - (pop(k).Position - Center) / m_obl;
        
        % Boundary Checking
        OppositePosition = max(OppositePosition, VarMin);
        OppositePosition = min(OppositePosition, VarMax);
        
        OppositeCost = CostFunction(OppositePosition);
        
        % Replace the worst individual if the new one is better
        if OppositeCost < pop(end).Cost
            pop(end).Position = OppositePosition;
            pop(end).Cost = OppositeCost;
            pop(end).GV = (pop(end).Position ./ (VarMax - VarMin));
        end
    end
    [~, SortOrder] = sort([pop.Cost]); % Re-sort after potential replacement
    pop = pop(SortOrder);
    
    % --- Asymmetric Perturbation Operator (t-distribution) ---
    if rand() < tk_pert
        t_r = exp(1 + (1 + it/MaxIt)^c2); % Eq. 11
        for k_pert = 1:nPop
            S = trnd(t_r, VarSize); % Random sample from t-distribution
            % Generate perturbed individual (Eq. 12)
            PerturbedPosition = Center + (pop(1).Position - Center) .* rand(VarSize) .* S;

            % Boundary Checking
            PerturbedPosition = max(PerturbedPosition, VarMin);
            PerturbedPosition = min(PerturbedPosition, VarMax);

            PerturbedCost = CostFunction(PerturbedPosition);

            % Update if better
            if PerturbedCost < pop(k_pert).Cost
                pop(k_pert).Position = PerturbedPosition;
                pop(k_pert).Cost = PerturbedCost;
                pop(k_pert).GV = (pop(k_pert).Position ./ (VarMax - VarMin));
            end
        end
        [~, SortOrder] = sort([pop.Cost]); % Re-sort after potential perturbations
        pop = pop(SortOrder);
    end

    % ----------------------------------------------------------------------
    % PHASE C: GLOBAL BEST UPDATE & STAGNATION CHECK (INNOVATION 2: ESCAPE MECHANISM)
    % ----------------------------------------------------------------------
    
    % Update Best Solution Ever Found
    if pop(1).Cost < BestSol.Cost
        BestSol = pop(1);
        stagnation_count = 0; % Reset stagnation counter
    else
        stagnation_count = stagnation_count + 1;
    end
    
    % Check for stagnation and apply escape mechanism
    if stagnation_count >= MaxStagnation
        
        % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % R6 FIX (Task 5): Elite Archive Sampling Bug 
        % The original code used max(2, ...), which causes an error in
        % randperm(n_elite, 3) when n_elite becomes 2.
        % Changed to max(3, ...) as suggested by R6.
        
        n_elite = max(3, ceil(nPop * p));
        n_elite = min(n_elite, nPop); % Safety cap: n_elite cannot be larger than nPop
        
        % --- End of R6 FIX (Task 5) Block ---
        % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        DEA = pop(1:n_elite);
        
        % Select a subset of non-best individuals to perturb
        n_escape = ceil(nPop / 2);
        
        % Ensure randperm is feasible if nPop-1 < n_escape
        if (nPop - 1) < n_escape
             n_escape = nPop - 1;
        end
        
        if n_escape > 0 % Only run if population is large enough
            escape_indices = randperm(nPop - 1, n_escape) + 1; % Indices from 2 to nPop
            
            for j = 1:numel(escape_indices)
                idx_to_escape = escape_indices(j);
                
                % Randomly select three distinct elites from DEA
                % This call is now safe thanks to the R6 FIX (Task 5)
                r_indices = randperm(n_elite, 3); 
                I_r1 = DEA(r_indices(1)).Position;
                I_r2 = DEA(r_indices(2)).Position;
                I_r3 = DEA(r_indices(3)).Position;
                
                % Generate Elite Differential Perturbation Vector (Eq. 14)
                V_edp = I_r1 + F * (I_r2 - I_r3);
                
                % Apply perturbation to the selected individual (Eq. 15)
                EscapedPosition = pop(idx_to_escape).Position + V_edp;

                % Boundary Checking
                EscapedPosition = max(EscapedPosition, VarMin);
                EscapedPosition = min(EscapedPosition, VarMax);
                
                EscapedCost = CostFunction(EscapedPosition);

                % Update if the new position is better
                if EscapedCost < pop(idx_to_escape).Cost
                    pop(idx_to_escape).Position = EscapedPosition;
                    pop(idx_to_escape).Cost = EscapedCost;
                    pop(idx_to_escape).GV = (pop(idx_to_escape).Position ./ (VarMax - VarMin));
                end
            end
        end
        
        [~, SortOrder] = sort([pop.Cost]); % Re-sort after escape mechanism
        pop = pop(SortOrder);
        stagnation_count = 0; % Reset stagnation counter
    end

    % Store Best Cost for Convergence Curve
    Convergence_curve(it) = BestSol.Cost;

    % Display Iteration Information
    disp(['Iteration ' num2str(it) ': Best Cost = ' num2str(Convergence_curve(it))]);

end

%% Results
BestSol_Cost = BestSol.Cost;
BestSol_Position = BestSol.Position;

end