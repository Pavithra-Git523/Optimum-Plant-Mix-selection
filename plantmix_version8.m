function [] = plantmix_version8()
clc;
clear;
tic
%%%%%%%%% Inputs %%%%%%%%%%%%%%%%%%%%%%%%%%%

plant_code=[1	2	3	4	5	6	7	8	9	10	11	12	13	14	15	16	17	18	19	20	21	22	23	24	25	26	27	28	29	30	31	32	33	34	35	36	37	38	39	40	41	42	43	44	45	46	47	48	49	50	51	52	53	54	55	56	57	58	59	60	61	62	63	64	65	66	67	68	69	70	71	72	73	74	75	76	77	78	79	80	81	82	83	84	85	86	87	88	89	90	91	92	93	94	95	96	97	98	99	100	101	102	103	104	105	106	107	108	109	110	111	112	113	114	115	116	117	118	119	120	121	122	123	124	125	126	127	128	129	130	131	132	133	134	135	136	137	138	139	140	141	142	143	144	145	146	147	148	149	150	151	152	153	154];% plant code
plant_selection=ones(1,154);% All the plants are selected
%plant_selection([10 20 30 40 50 60 70 80 90 100 155])=[0 0 0 0 0 0 0 0 0 0 0];% omit some plants
value=[6	6	4	3	6	6	3	5	6	7	4	6	5	6	5	3	6	5	6	5	3	5	8	8	4	5	4	7	7	5	4	5	4	3	5	3	4	4	5	4	7	4	5	3	7	7	8	8	5	5	5	3	5	5	5	5	6	4	3	5	5	6	5	4	7	6	4	7	6	4	4	3	3	7	3	5	4	6	7	7	7	5	5	7	8	5	6	5	6	4	5	4	2	5	3	3	9	5	5	3	3	8	8	5	4	7	6	6	7	9	3	5	5	7	5	3	7	7	6	6	5	6	5	6	6	8	8	7	6	7	9	9	10	6	7	10	7	6	5	9	4	6	6	7	4	5	6	6	6	7	8	5	5	7]; 
space_requirement=[0.0125	0.0125	0.0125	0.01	0.0225	0.0256	0.0192	0.03	0.05	0.0625	0.05	0.12	0.1125	0.16	0.16	0.0968	0.2	0.18	0.225	0.24	0.125	0.18	0.36	0.36	0.18	0.25	0.2025	0.36	0.36	0.27	0.225	0.36	0.25	0.2	0.36	0.36	0.3125	0.3125	0.45	0.36	0.5625	0.4332	0.5625	0.36	0.75	0.9	1	1	0.72	0.75	0.75	0.45	0.81	0.81	0.81	0.8281	1	0.72	0.5625	1	1	1	1	0.81	1.44	1.08	0.9	1.44	1.44	1	1.08	0.81	0.81	2.025	0.9	1.5	1.215	1.92	2.25	2.25	2.25	1.62	1.96	2.43	2.88	2	2.43	2.025	2.25	1.8	2.5	2	1.0125	3.24	2	2.025	6.25	4	4	2.43	1.5	9	9	6.48	7.2	9	9	9	9	12.96	5.76	10	11.25	16	11.52	8	16	25	20.25	25	18	25	25	25	25	36	36	36	36	36	49	49	64	36	49	81	56.25	64	44.89	81	36	64	64	81	44.89	64	100	100	100	121	144	100	144	225];
minimum_num_of_plants=[5	5	5	4	1	4	3	3	5	1	5	3	5	1	1	2	20	2	5	6	2	4	1	1	2	1	1	1	1	3	5	1	1	5	1	1	5	5	5	1	1	3	1	1	3	10	1	1	2	3	3	5	1	1	1	1	1*1	2	1	1*1	1*1	1	1*1	1	1	1	10	1	1	1	1	1	1	10	10	1*1.5	6	3	1	1	1	2	1	3	2	2	3	10	1	20	10	1	5	1	2	10	1	1	1	3	25	1	1	2	5	1	1	1	1	1	1	2	5	1	2	2	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1];


Available_Area=75;
max_itr=100;
max_run=50;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

xx=[plant_code;plant_selection;value;space_requirement;minimum_num_of_plants];
dataMat = xx'

secondcolum=dataMat(:,2);
A0=dataMat(secondcolum==1,:);

plant_representation = A0(:,2);% selected plants
spacerequirement = A0(:,4);
n = 2*size(A0,1);% Initial Population Size
m=size(A0,1);% Number of selected varities


%%%%%%%%%%%%%%%%%%Plot%%%%%%%%%%%%%%%%%%%

% store best fit,count for best fit and area for each iteration
itr_fitness_mat=zeros(max_itr,1);
itr_count_of_varities=zeros(max_itr,1);
itr_area_mat=zeros(max_itr,1);

%store best fit,count for best fit and area for each run
Best_fitness_for_runs=zeros(max_run,1);
Varities_count_for_runs=zeros(max_run,1);
Best_area_for_runs=zeros(max_run,1);
Selected_chromosomes_for_runs=zeros(max_run,m);

%store best fit,count for best fit and area for each iteration in each run
A=zeros(max_run,max_itr);
B=zeros(max_run,max_itr);
C=zeros(max_run,max_itr);



figure(1); % new figure
s(1) = subplot(3,1,1); % top subplot
s(2) = subplot(3,1,2); % middle subplot
s(3) =subplot(3,1,3);% bottom subplot


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%________________Termination criteria____________________%
%%%%%%%%%%%%%%%%%%%%%% runs %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for r=1:max_run
    
[plant_mix]=Initial_Population_selection(n,m,A0);% Initial Population Selection
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%% iterations %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for itr=1:max_itr
    
    [fitness plantingarea] = evaluateFitness(plant_mix,A0,Available_Area,n,m);%Evaluate fitness of each planting mix
    [Best_fitness_for_itr ind]= max(fitness);
    no_of_varities = nnz(plant_mix(ind,:)); % count of the varities for each iteration
    Area=plantingarea(ind);

crossedChromo = doCrossover(plant_mix,fitness,n,m);%crossing over
[mutatedChromo] = doMutation(crossedChromo,plantingarea,spacerequirement,Available_Area,n,m);% doing mutation

[Chromofit newplantingarea] = evalFit(mutatedChromo,A0,Available_Area,n,m);
[Minfit minfit_ind]=min(Chromofit);
mutatedChromo(minfit_ind,:)=plant_mix(ind,:); %include elitism
plant_mix = mutatedChromo;

% store values in each iteration
itr_fitness_mat(itr,1)=Best_fitness_for_itr; %fitness
itr_count_of_varities(itr,1)=no_of_varities; %count of varities
itr_area_mat(itr,1)=Area; % area

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%% Output of last iteration in each run %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[fitness plantingarea] = evaluateFitness(plant_mix,A0,Available_Area,n,m);%finding fitness of the final population
[Best_fitness_final ind]= max(fitness);% finding maximum fitness
final_Area=plantingarea(ind);
Selection=plant_mix(ind,:);
Selected_varites_count_final=nnz(Selection);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%% store values of each run %%%%%%%%%%%%%%%%%%%
Best_fitness_for_runs(r,1)=Best_fitness_final;
Varities_count_for_runs(r,1)=Selected_varites_count_final;
Best_area_for_runs(r,1)=final_Area;
Selected_chromosomes_for_runs(r,:)=Selection;
A(r,:)=itr_fitness_mat';% fitness values of each iteration for each run
B(r,:)=itr_count_of_varities'; % count of varities for each iteration in each run
C(r,:)=itr_area_mat'; %planting area of each iteration for each run

%%%%%%%%%%%%%%% Display %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if r==max_run
disp('_______________display selected varities,number of varities and required area for last run_______________')

Plant_Code=A0(:,1).';
Plantmix=[Plant_Code;Selected_chromosomes_for_runs(r,:)];
secondrow=Plantmix(2,:);
Selected_varities=Plantmix(:,secondrow==1);
[member,plant_selection] = ismember(Selected_varities(1,:),xx(1,:));% find where the codes match in the 1st row
plant_selection = plant_selection(member);% save only the column numbers where the codes match
Selected_varities(2,:)=xx(5,plant_selection);% replace the columns where the codes match
varity=Selected_varities;
array2table(varity,'RowNames',{'Plant Code','number of plants'})
plot(s(1),A(r,:),'-')
        %title(s(1),'Best fitness values against the iterations for last run')
        ylabel(s(1),'Fitness')
        xlabel(s(1),'Iterations')
        
        plot(s(2),B(r,:),'g-')
        %title(s(2),'Count of varieties against the iterations for last run')
        ylabel(s(2),'Varieties count')
        xlabel(s(2),'Iterations')
    
        plot(s(3),C(r,:),'k-')
        %title(s(3),'Best plantation area against the iterations for last run')
        ylabel(s(3),'Plantation area')
        xlabel(s(3),'Iterations')
        
        Best_fitness_of_lastrun=Best_fitness_for_runs(r,1)
        varities_count_of_last_run=Varities_count_for_runs(r,1)
        PlantationArea_for_last_run=Best_area_for_runs(r,1)   
        
end 

     
end 
 
figure(2); % new figure
u(1) = subplot(3,1,1); % top subplot
u(2) = subplot(3,1,2); % middle subplot
u(3) = subplot(3,1,3); % bottom subplot
%%%%%%%%%%%%%%% Display %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('_______________display selected varities,number of selected varities and required area for best run_______________')

[best_fit ind1]=max(Best_fitness_for_runs);

Plant_Code=A0(:,1).';
Plantmix=[Plant_Code;Selected_chromosomes_for_runs(ind1,:)];
secondrow=Plantmix(2,:);
Selected_varities=Plantmix(:,secondrow==1);
[member,plant_selection] = ismember(Selected_varities(1,:),xx(1,:));% find where the codes match in the 1st row
plant_selection = plant_selection(member);% save only the column numbers where the codes match
Selected_varities(2,:)=xx(5,plant_selection);% replace the columns where the codes match
varity=Selected_varities;
array2table(varity,'RowNames',{'Plant Code','number of plants'})
plot(u(1),A(ind1,:),'-')
       % title(u(1),'Best fitness values against the iterations for best run')
        ylabel(u(1),'Fitness ')
        xlabel(u(1),'Iterations')
        
        plot(u(2),B(ind1,:),'g-')
        %title(u(2),'Count of varities against the iterations for best run')
        ylabel(u(2),'Varieties count')
        xlabel(u(2),'Iterations')
    
        plot(u(3),C(ind1,:),'k-')
        %title(u(3),'Best plantation area against the iterations for best run')
        ylabel(u(3),'Plantation area')
        xlabel(u(3),'Iterations')
        
        fitness_of_bestfit_run=Best_fitness_for_runs(ind1,1)
        varities_count_of_bestfit_run=Varities_count_for_runs(ind1,1)
        PlantationArea_for_bestfit_run=Best_area_for_runs(ind1,1)


        
        % fitness value and area of selected chromosome for each run
        disp('__________________display number of varities and area of each run________________')
        
[Best_fitness_for_each_run]=Best_fitness_for_runs'
[Varities_count_for_each_run]=Varities_count_for_runs'
[Area_for_each_run]=Best_area_for_runs'
Selected_chromosomes_for_runs;
Maximum=max(Best_fitness_for_each_run)
Minimum=min(Best_fitness_for_each_run)
Average=mean(Best_fitness_for_each_run)

figure(3);
m(1) = subplot(3,1,1); % top subplot
m(2) = subplot(3,1,2); % middle subplot
m(3) = subplot(3,1,3); %bottom subplot
        
% Plot of Finess values and area against each run
plot(m(1),Best_fitness_for_runs,'-')
%title(m(1),'Fitness value againts the runs')
 ylabel(m(1),'Best Fitness')
 xlabel(m(1), 'Runs')
 
 plot(m(2),Varities_count_for_runs,'g-');
 %title(m(2),'Number of selected varities against the runs')
  ylabel(m(2),'Varieties Count')
  xlabel(m(2),'Runs')
  
plot(m(3),Best_area_for_runs,'k-');
 %title(m(3),'Area against the runs')
  ylabel(m(3),'Area')
  xlabel(m(3),'Runs')
  
  toc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%Initial Population Selection Function with sub functions%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function[plant_mix]=Initial_Population_selection(n,m,A0);
[Initial_Plant_Mix] = init_pop(n,m);
[Selection_Prob] = evalFitness(A0,n,m);
[new_Plant_Mix] = Select(Initial_Plant_Mix,Selection_Prob,n,m);
plant_mix =new_Plant_Mix ;

function [Initial_Plant_Mix] = init_pop(n,m)%creating initial population in the binary format
for i=1:n
Initial_Plant_Mix(i,:) = zeros(1,m);
end



function [Selection_Prob] = evalFitness(A0,n,m)
value = A0(:,3);
spacerequirement = A0(:,4);
    
    for k=1:m 
        fit(k) = (value(k)./spacerequirement(k));%getting the importance value of each plant
    end
    [fitVal] = fit;
 
    rank =floor(tiedrank(fitVal)); %rank according to importance value
    sumOfRanks = sum(rank);
    [Selection_Prob]= rank./sumOfRanks; % Probability of selcting each plant varity to the planting mix
   
    function [new_Plant_Mix] = Select(Initial_Plant_Mix,Selection_Prob,n,m)
for i=1:n
    for j = 1:m
        if(rand()<=Selection_Prob(j))
            Initial_Plant_Mix(i,j) = 1;
        else
           Initial_Plant_Mix(i,j) = Initial_Plant_Mix(i,j); 
        end
    end
    new_Plant_Mix(i,:) = Initial_Plant_Mix(i,:);
end
   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%% Maximize the number of Varities %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Function for evaluate Fitness
function [fitness plantingarea] = evaluateFitness(plant_mix,A0,Available_Area,n,m)
spacerequirement = A0(:,4);
for j=1:n
    sum = 0;
    sumSpacerequirement = 0;
    for k=1:m %length_initialselection
        sum = sum +(plant_mix(j,k)); %get the count of varities
        sumSpacerequirement = sumSpacerequirement + (plant_mix(j,k).*spacerequirement(k)); %getting the total planting area of a varities
    end
    fitness(j) = sum;
    plantingarea(j) = sumSpacerequirement;
   %calculating fitness according to the Available Area
    if plantingarea(j)<=Available_Area
        fitness(j) = sum;
    else
        fitness(j) = 0.1*sum;
    end
end


% rank selection to select a perticular chromosome of varities
function[selectedVarities] = doRankSelection (plant_mix,fitness,n,m)

[fitnessVal indexVal] = sort(fitness);

convertedChromo = plant_mix(indexVal,:);

rank =(1:1:m);
sumOfRanks = sum(rank);

randomVal = sumOfRanks*rand();
    sumVal=0;
    for gg=1:n
        for i=1:m
        sumVal = sumVal + rank(i);
        if (sumVal >= randomVal)
            selectedChromosomePosition = gg;
        end
        end
       
    end
selectedVarities = convertedChromo(selectedChromosomePosition,:);


% Do crossover to find better planting mix
function[crossedChromo] = doCrossover(plant_mix,fitness,n,m)
for i = 1:2:n
parent1  =  doRankSelection (plant_mix,fitness,n,m);
parent2 =  doRankSelection (plant_mix,fitness,n,m);

randProbofCross = 0.95;

if rand()<=randProbofCross %(crossing)
 pointToCross = randi(m-1);%selecting a position randomly as the crossover point
 C1part1 = parent1(1:pointToCross);
 C1part2 = parent2(pointToCross+1:m);
 child1 = [C1part1 C1part2];
 
 C2part1 = parent2(1:pointToCross);
 C2part2 = parent1(pointToCross+1:m);
 child2 = [C2part1 C2part2];

else%(Cloning)
child1 = parent1;
child2 = parent2;
end

crossedChromo (i,:) = child1;
crossedChromo (i+1,:) = child2;
end

% Do mutation to explore better solutions
function[mutatedChromo] = doMutation(crossedChromo,plantingarea,spacerequirement,Available_Area,n,m)
muteProb = 0.05;
for i=1:n
    for j = 1:m
        if(rand()<=muteProb)
                
                    if(crossedChromo(i,j)==0)
                        crossedChromo(i,j) = 1;
                    else
                        crossedChromo(i,j) = 0;
                    end
                
        else
           crossedChromo(i,j) = crossedChromo(i,j); 
        end
    end
    mutatedChromo(i,:) = crossedChromo(i,:);
 
end

% Function for evaluate Fitness of mutated Chromosome
function [Chromofit newplantingarea] = evalFit(mutatedChromo,A0,Available_Area,n,m)
spacerequirement = A0(:,4);
for j=1:n
    sum = 0;
    sumSpacerequirement = 0;
    for k=1:m %length_initialselection
        sum = sum +(mutatedChromo(j,k)); %get the count of varities
        sumSpacerequirement = sumSpacerequirement + (mutatedChromo(j,k).*spacerequirement(k)); %getting the total planting area of a varities
    end
    Chromofit(j) = sum;
    newplantingarea(j) = sumSpacerequirement;
   %calculating fitness according to the Available Area
    if newplantingarea(j)<=Available_Area
        Chromofit(j) = sum;
    else
        Chromofit(j) = 0.1*sum;
    end
end

