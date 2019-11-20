/*******************************************************************************      
Program Name: 	2013-0533_do_tables  
Contact:  		Cynthia Kinnan (c-kinnan@northwestern.edu)
Last Modified: 	5 May 2014
Purpose: 		Replicates all tables from "The miracle of microfinance? Evidence
				from a randomized evaluation" (Banerjee et al.), AEJ, 2014
Files Used: 	2013-0533_data_baseline.dta
				2013-0533_data_endlines1and2.dta
				2013-0533_data_census.dta
				2013-0533_data_endline1businesstype.dta
Files Created:	table1a.txt
				table1b.txt
				table2.txt
				table3.txt
				table3b.txt
				table3c.txt
				table4.txt
				table5.txt
				table6.txt
				table7.txt
				table_index_pvals.txt
				tableA1.txt
				tableA2.txt
				tableA3.txt
				tableA4.txt
				tableA5.txt
*******************************************************************************/
cap log close
clear all
set more off
set mem 100m
pause on

*DATA DIRECTORY
global datadir "C:/Users/hreppst/Dropbox/Spandana/Paper/AEJ Final/Data/"

*OUTPUT DIRECTORY
global outputdir "C:/Users/hreppst/Dropbox/Spandana/Paper/AEJ Final/Data/Output"

*LOG FILE
log using 2013-0533_log_tables.smcl, replace 

*CONTROLS FOR BASELINE VARIABLES
global area_controls "area_pop_base area_debt_total_base area_business_total_base area_exp_pc_mean_base area_literate_head_base area_literate_base"

cd "$outputdir"

********************************************************************************
*************        Table 1A: Baseline Summary Statistics        **************
********************************************************************************
use "$datadir/2013-0533_data_baseline.dta", clear

local hh_composition "hh_size adult children male_head head_age head_noeduc"
local credit_access "spandana othermfi bank informal anyloan"
local loan_amt "spandana_amt othermfi_amt bank_amt informal_amt anyloan_amt"
local self_emp_activ "total_biz female_biz female_biz_pct"
local businesses "bizrev bizexpense bizinvestment bizemployees hours_weekbiz"
local businesses_allHH "bizrev_allHH bizexpense_allHH bizinvestment_allHH bizemployees_allHH hours_weekbiz_allHH"
local consumption "total_exp_mo nondurable_exp_mo durables_exp_mo home_durable_index"

local allvars "`hh_composition' `credit_access' `loan_amt' `self_emp_activ' `businesses' `businesses_allHH' `consumption'"


*1) Generate variables for business outcomes for all households (replacing each
*	variable with 0 for households without a business):
foreach var in `businesses' {
	gen `var'_allHH=`var'
	replace `var'_allHH=0 if total_biz==0
	}

*2) Create summary statistics table:
scalar drop _all
foreach var of varlist `allvars' {
	sum `var' if treatment==0
	scalar `var'_N=r(N)
	scalar `var'_c=r(mean)
	scalar sd`var'_c=r(sd)
	reg `var' treatment, cluster(areaid)
	mat beta`var'=e(b)
	scalar `var'_d=beta`var'[1,1]
	test treatment=0
	scalar pval_`var'=r(p)
	matrix Table1A = [nullmat(Table1A) \ `var'_N, `var'_c, sd`var'_c, `var'_d, pval_`var']
	}
matrix rownames Table1A = `allvars'
matrix colnames Table1A = Obs Control_mean Control_sd Difference p_val
mat2txt, matrix(Table1A) saving("$outputdir/table1a.txt") replace


********************************************************************************
*************         Table 1B: Endline Summary Statistics        **************
********************************************************************************
use "$datadir/2013-0533_data_endlines1and2.dta", clear

local hh_composition "hhsize adults children male_head head_age head_noeduc"
local credit_access "spandana othermfi anybank anyinformal anyloan"
local loan_amt "spandana_amt othermfi_amt bank_amt informal_amt anyloan_amt"
local self_emp_activ "total_biz female_biz_allHH female_biz_pct"
local businesses "bizrev bizexpense bizinvestment bizemployees hours_week_biz"
local businesses_allHH "bizrev_allHH bizexpense_allHH bizinvestment_allHH bizemployees_allHH hours_week_biz_allHH"
local consumption "total_exp_mo nondurable_exp_mo durables_exp_mo home_durable_index"

local allvars "`hh_composition' `credit_access' `loan_amt' `self_emp_activ' `businesses' `businesses_allHH' `consumption'"


*1) Generate variables for business outcomes for all households (replacing each
*	variable with missing for households without a business):
forval i = 1/2 {
	foreach var in `businesses' female_biz {
			gen `var'_allHH_`i'=`var'_`i'
			replace `var'_`i'=. if total_biz_`i'==0
			}
	}

*2) Reshape to one observation per household per endline:
foreach var in `allvars' {
	rename `var'_1 `var'1
	rename `var'_2 `var'2
	}

reshape long `allvars', i(hhid) j(endline)
keep hhid areaid endline treatment `allvars'
tab endline, gen(endline)

*3) Create summary statistics table:
scalar drop _all
foreach var of varlist `allvars' {
	sum `var' if treatment==0 & endline==1
	scalar `var'_N_el1=r(N)
	scalar `var'_c_el1=r(mean)
	scalar sd`var'_c_el1=r(sd)
	
	sum `var' if treatment==0 & endline==2
	scalar `var'_N_el2=r(N)
	scalar `var'_c_el2=r(mean)
	scalar sd`var'_c_el2=r(sd)

	reg `var' endline2 if treatment==0, cluster(areaid)
	mat beta`var'=e(b)
	scalar `var'_d=beta`var'[1,1]
	test endline2=0
	scalar pval_`var'=r(p)
	
	matrix Table1B = [nullmat(Table1B) \ `var'_N_el1, `var'_c_el1, sd`var'_c_el1, `var'_N_el2, `var'_c_el2, sd`var'_c_el2, `var'_d, pval_`var']
	}
matrix rownames Table1B = `allvars'
matrix colnames Table1B = Obs_el1 Control_mean_el1 Control_sd_el1 Obs_el2 Control_mean_el2 Control_sd_el2 Difference p_val
mat2txt, matrix(Table1B) saving("$outputdir/table1b.txt") replace


********************************************************************************
*************                   Table 2: Credit                   **************
********************************************************************************
use "$datadir/2013-0533_data_endlines1and2.dta", clear

*PANEL A: ENDLINE 1
est clear
foreach var in spandana_1 othermfi_1 anymfi_1 anybank_1	anyinformal_1  anyloan_1 everlate_1 mfi_loan_cycles_1 ///
			   spandana_amt_1 othermfi_amt_1 anymfi_amt_1 bank_amt_1 informal_amt_1 anyloan_amt_1 credit_index_1 {
	reg `var' treatment $area_controls [pweight=w1], cluster(areaid)
	eret2 scalar pval=2*ttail(e(df_r),abs(_b[treatment]/_se[treatment]))
	sum `var' if treatment==0 & e(sample)
	eret2 scalar mn1=r(mean)
	eret2 scalar sd1=r(sd)
	est store `var'
}
estout * using "table2.txt", drop($area_controls _cons) title("Table 2: Credit, Endline 1") ///
	prehead("" @title) cells(b(fmt(a3) s) se(fmt(a3) par(`"="("'`")""'))) replace s(r2 mn1 sd1 N pval) ///
	starlevels(* .1 ** .05 *** .01) legend ///
	postfoot("Robust standard errors, clustered at the area level, in brackets.")

*PANEL B: ENDLINE 2
est clear
foreach var in spandana_2 othermfi_2 anymfi_2 anybank_2	anyinformal_2 anyloan_2 everlate_2 mfi_loan_cycles_2 ///
			   spandana_amt_2 othermfi_amt_2 anymfi_amt_2 bank_amt_2 informal_amt_2 anyloan_amt_2 credit_index_2 {
	reg `var' treatment $area_controls [pweight=w2], cluster(areaid)
	eret2 scalar pval=2*ttail(e(df_r),abs(_b[treatment]/_se[treatment]))
	sum `var' if treatment==0 & e(sample)
	eret2 scalar mn2=r(mean)
	eret2 scalar sd2=r(sd)
	est store `var'
}
estout * using "table2.txt", drop($area_controls _cons)  title("Table 2: Credit, Endline 2")	///
	prehead("" @title) cells(b(fmt(a3) s) se(fmt(a3) par(`"="("'`")""'))) append s(r2 mn2 sd2 N pval) ///
	starlevels(* .1 ** .05 *** .01)	legend ///
	postfoot("Robust standard errors, clustered at the area level, in brackets.")


********************************************************************************
****         Table 3: Self-employment activities (all households)          *****
********************************************************************************
use "$datadir/2013-0533_data_endlines1and2.dta", clear

*PANEL A: ENDLINE 1
est clear
foreach var in bizassets_1 bizinvestment_1 bizrev_1	bizexpense_1 bizprofit_1 any_biz_1 ///	
			   total_biz_1 any_new_biz_1 biz_stop_1 newbiz_1 female_biz_new_1 biz_index_all_1 {
	reg `var' treatment $area_controls [pweight=w1], cluster(areaid)
	eret2 scalar pval=2*ttail(e(df_r),abs(_b[treatment]/_se[treatment]))
	sum `var' if treatment==0 & e(sample)
	eret2 scalar mn1=r(mean)
	eret2 scalar sd1=r(sd)
	est store `var'
}
estout * using "table3.txt", drop($area_controls _cons) title("Table 3: Self-employment activities, Endline 1") ///
	prehead("" @title) cells(b(fmt(a3) s) se(fmt(a3) par(`"="("'`")""'))) replace s(r2 mn1 sd1 N pval) ///
	starlevels(* .1 ** .05 *** .01) legend ///
	postfoot("Robust standard errors, clustered at the area level, in brackets.")	

*PANEL B: ENDLINE 2
est clear
foreach var in bizassets_2 bizinvestment_2 bizrev_2	bizexpense_2 bizprofit_2 any_biz_2 ///	
			   total_biz_2 any_new_biz_2 biz_stop_2 newbiz_2 female_biz_new_2 biz_index_all_2 {
	reg `var' treatment $area_controls [pweight=w2], cluster(areaid)
	eret2 scalar pval=2*ttail(e(df_r),abs(_b[treatment]/_se[treatment]))
	sum `var' if treatment==0 & e(sample)
	eret2 scalar mn2=r(mean)
	eret2 scalar sd2=r(sd)
	est store `var'
}
estout * using "table3.txt", drop($area_controls _cons) title("Table 3: Self-employment activities, Endline 2") ///
	prehead("" @title) cells(b(fmt(a3) s) se(fmt(a3) par(`"="("'`")""'))) append s(r2 mn2 sd2 N pval) ///
	starlevels(* .1 ** .05 *** .01) legend ///
	postfoot("Robust standard errors, clustered at the area level, in brackets.")	

	
********************************************************************************
***   Table 3: Self-employment activities (households with old businesses)   ***
********************************************************************************	
use "$datadir/2013-0533_data_endlines1and2.dta", clear

*PANEL A: ENDLINE 1
est clear
foreach var in bizassets_1 bizinvestment_1 bizrev_1 bizexpense_1 bizprofit_1 bizemployees_1 biz_index_old_1 {
	reg `var' treatment $area_controls if any_old_biz==1 [pweight=w1], cluster(areaid)
	eret2 scalar pval=2*ttail(e(df_r),abs(_b[treatment]/_se[treatment]))
	sum `var' if treatment==0 & any_old_biz==1
	eret2 scalar mn1=r(mean)
	eret2 scalar sd1=r(sd)
	est store `var'
}
estout * using "table3b.txt", drop($area_controls _cons) title("Table 3: Self-employment activities (households with old businesses), Endline 1") ///
	prehead("" @title) cells(b(fmt(a3) s) se(fmt(a3) par(`"="("'`")""'))) replace s(r2 mn1 sd1 N pval) ///
	starlevels(* .1 ** .05 *** .01) legend ///
	postfoot("Robust standard errors, clustered at the area level, in brackets.")	

*PANEL B: ENDLINE 2
est clear
foreach var in bizassets_2 bizinvestment_2 bizrev_2 bizexpense_2 bizprofit_2 bizemployees_2 biz_index_old_2 {
	reg `var' treatment $area_controls if any_old_biz==1 [pweight=w2], cluster(areaid)
	eret2 scalar pval=2*ttail(e(df_r),abs(_b[treatment]/_se[treatment]))
	sum `var' if treatment==0 & any_old_biz==1
	eret2 scalar mn2=r(mean)
	eret2 scalar sd2=r(sd)
	est store `var'
}
estout * using "table3b.txt", drop($area_controls _cons) title("Table 3: Self-employment activities (households with old businesses), Endline 2") ///
	prehead("" @title) cells(b(fmt(a3) s) se(fmt(a3) par(`"="("'`")""'))) append s(r2 mn2 sd2 N pval) ///
	starlevels(* .1 ** .05 *** .01) legend ///
	postfoot("Robust standard errors, clustered at the area level, in brackets.")	


********************************************************************************
** Table 3: Self-employment activities (households with new businesses, EL1)  **
********************************************************************************
use "$datadir/2013-0533_data_endlines1and2.dta", clear

est clear
foreach var in bizassets_1 bizinvestment_1 bizrev_1 bizexpense_1 bizprofit_1 bizemployees_1 biz_index_new_1 {
	reg `var' treatment $area_controls if newbiz_1>0 & newbiz_1!=. [pweight=w1], cluster(areaid)
	eret2 scalar pval=2*ttail(e(df_r),abs(_b[treatment]/_se[treatment]))
	sum `var' if treatment==0 & newbiz_1>0 & newbiz_1!=. 
	eret2 scalar mn1=r(mean)
	eret2 scalar sd1=r(sd)
	est store `var'
}
estout * using "table3c.txt", drop($area_controls _cons) title("Table 3: Self-employment activities (households with new businesses), Endline 1") ///
	prehead("" @title) cells(b(fmt(a3) s) se(fmt(a3) par(`"="("'`")""'))) replace s(r2 mn1 sd1 N pval) ///
	starlevels(* .1 ** .05 *** .01) legend ///
	postfoot("Robust standard errors, clustered at the area level, in brackets.")	


********************************************************************************
*************                   Table 4: Income                   **************
********************************************************************************
use "$datadir/2013-0533_data_endlines1and2.dta", clear

*PANEL A: ENDLINE 1
est clear
foreach var in bizprofit_1 wages_nonbiz_1 income_index_1 {
	reg `var' treatment $area_controls [pweight=w1], cluster(areaid)
	eret2 scalar pval=2*ttail(e(df_r),abs(_b[treatment]/_se[treatment]))
	sum `var' if treatment==0 & e(sample)
	eret2 scalar mn1=r(mean)
	eret2 scalar sd1=r(sd)
	est store `var'
}
estout * using "table4.txt", drop($area_controls _cons) title("Table 4: Income, Endline 1") ///
	prehead("" @title) cells(b(fmt(a3) s) se(fmt(a3) par(`"="("'`")""'))) replace s(r2 mn1 sd1 N pval) ///
	starlevels(* .1 ** .05 *** .01) legend ///
	postfoot("Robust standard errors, clustered at the area level, in brackets.")

*PANEL B: ENDLINE 2
est clear
foreach var in bizprofit_2 wages_nonbiz_2 income_index_2 {
	reg `var' treatment $area_controls [pweight=w2], cluster(areaid)
	eret2 scalar pval=2*ttail(e(df_r),abs(_b[treatment]/_se[treatment]))
	sum `var' if treatment==0 & e(sample)
	eret2 scalar mn2=r(mean)
	eret2 scalar sd2=r(sd)
	est store `var'
}
estout * using "table4.txt", drop($area_controls _cons) title("Table 4: Income, Endline 2") ///
	prehead("" @title) cells(b(fmt(a3) s) se(fmt(a3) par(`"="("'`")""'))) append s(r2 mn2 sd2 N pval) ///
	starlevels(* .1 ** .05 *** .01) legend ///
	postfoot("Robust standard errors, clustered at the area level, in brackets.")


********************************************************************************
*********                Table 5: Household labor hours               **********
********************************************************************************
use "$datadir/2013-0533_data_endlines1and2.dta", clear

*PANEL A: ENDLINE 1
est clear
foreach var in hours_week_1 hours_week_biz_1 hours_week_outside_1 ///
			   hours_girl1620_week_1 hours_boy1620_week_1 ///
               hours_headspouse_week_1 hours_headspouse_biz_1 hours_headspouse_outside_1 labor_index_1 {
	reg `var' treatment $area_controls [pweight=w1], cluster(areaid)
	eret2 scalar pval=2*ttail(e(df_r),abs(_b[treatment]/_se[treatment]))	
	sum `var' if treatment==0 & e(sample)
	eret2 scalar mn1=r(mean)
	eret2 scalar sd1=r(sd)
	est store `var'
}
estout * using "table5.txt", drop($area_controls _cons) title("Table 5: Time worked by household members, Endline 1") ///
	prehead("" @title) cells(b(fmt(a3) s) se(fmt(a3) par(`"="("'`")""'))) replace s(r2 mn1 sd1 N pval) ///
	starlevels(* .1 ** .05 *** .01) legend ///
	postfoot("Robust standard errors, clustered at the area level, in brackets.")
	
*PANEL B: ENDLINE 2
est clear
foreach var in hours_week_2 hours_week_biz_2 hours_week_outside_2 ///
			   hours_girl1620_week_2 hours_boy1620_week_2 ///
                hours_headspouse_week_2 hours_headspouse_biz_2 hours_headspouse_outside_2 labor_index_2 {
	reg `var' treatment $area_controls [pweight=w2], cluster(areaid)
	eret2 scalar pval=2*ttail(e(df_r),abs(_b[treatment]/_se[treatment]))	
	sum `var' if treatment==0 & e(sample)
	eret2 scalar mn2=r(mean)
	eret2 scalar sd2=r(sd)
	est store `var'
}
estout * using "table5.txt", drop($area_controls _cons) title("Table 5: Time worked by household members, Endline 2") ///
	prehead("" @title) cells(b(fmt(a3) s) se(fmt(a3) par(`"="("'`")""'))) append s(r2 mn2 sd2 N pval) ///
	starlevels(* .1 ** .05 *** .01) legend ///
	postfoot("Robust standard errors, clustered at the area level, in brackets.")


********************************************************************************
***********                   Table 6: Consumption                  ************
********************************************************************************
use "$datadir/2013-0533_data_endlines1and2.dta", clear

*PANEL A: ENDLINE 1
est clear
foreach var in total_exp_mo_pc_1 durables_exp_mo_pc_1 nondurable_exp_mo_pc_1 food_exp_mo_pc_1 health_exp_mo_pc_1 ///
			   educ_exp_mo_pc_1 temptation_exp_mo_pc_1 festival_exp_mo_pc_1 home_durable_index_1 {
	reg `var' treatment $area_controls [pweight=w1], cluster(areaid) level(90)
	eret2 scalar pval=2*ttail(e(df_r),abs(_b[treatment]/_se[treatment]))	
	sum `var' if treatment==0 & e(sample)
	eret2 scalar mn1=r(mean)
	eret2 scalar sd1=r(sd)
	est store `var'
}
estout * using "table6.txt", drop($area_controls _cons) title("Table 6: Consumption, Endline 1") ///
	prehead("" @title) cells(b(fmt(a3) s) se(fmt(a3) par(`"="("'`")""'))) replace s(r2 mn1 sd1 N pval) ///
	starlevels(* .1 ** .05 *** .01) legend ///
	postfoot("Robust standard errors, clustered at the area level, in brackets.")

*PANEL B: ENDLINE 2
est clear
foreach var in total_exp_mo_pc_2 durables_exp_mo_pc_2 nondurable_exp_mo_pc_2 food_exp_mo_pc_2 health_exp_mo_pc_2 ///
			   educ_exp_mo_pc_2 temptation_exp_mo_pc_2 festival_exp_mo_pc_2 home_durable_index_2 {
	reg `var' treatment $area_controls [pweight=w2], cluster(areaid)
	eret2 scalar pval=2*ttail(e(df_r),abs(_b[treatment]/_se[treatment]))	
	sum `var' if treatment==0 & e(sample)
	eret2 scalar mn2=r(mean)
	eret2 scalar sd2=r(sd)
	est store `var'
}
estout * using "table6.txt", drop($area_controls _cons) title("Table 6: Consumption, Endline 2") ///
	prehead("" @title) cells(b(fmt(a3) s) se(fmt(a3) par(`"="("'`")""'))) append s(r2 mn2 sd2 N pval) ///
	starlevels(* .1 ** .05 *** .01) legend ///
	postfoot("Robust standard errors, clustered at the area level, in brackets.")


********************************************************************************
**********                  Table 7: Social effects                  ***********
********************************************************************************
use "$datadir/2013-0533_data_endlines1and2.dta", clear

*PANEL A: ENDLINE 1
est clear
foreach var in girl515_school_1 boy515_school_1 girl515_workhrs_pc_1 boy515_workhrs_pc_1 girl1620_school_1 boy1620_school_1 ///
			   women_emp_index_1 female_biz_new_1 social_index_1 {
	reg `var' treatment $area_controls [pweight=w1], cluster(areaid)
	eret2 scalar pval=2*ttail(e(df_r),abs(_b[treatment]/_se[treatment]))	
	sum `var' if treatment==0 & e(sample)
	eret2 scalar mn1=r(mean)
	eret2 scalar sd1=r(sd)
	est store `var'
}
estout * using "table7.txt", drop($area_controls _cons) title("Table 7: Social effects, Endline 1") ///
	prehead("" @title) cells(b(fmt(a3) s) se(fmt(a3) par(`"="("'`")""'))) replace s(r2 mn1 sd1 N pval) ///
	starlevels(* .1 ** .05 *** .01) legend ///
	postfoot("Robust standard errors, clustered at the area level, in brackets.")

*PANEL B: ENDLINE 2
est clear
foreach var in girl515_school_2 boy515_school_2 girl515_workhrs_pc_2 boy515_workhrs_pc_2 girl1620_school_2 boy1620_school_2 ///
			   women_emp_index_2 female_biz_pct_2 female_biz_new_2 social_index_2 {
	reg `var' treatment $area_controls [pweight=w2], cluster(areaid)
	eret2 scalar pval=2*ttail(e(df_r),abs(_b[treatment]/_se[treatment]))	
	sum `var' if treatment==0&e(sample)
	eret2 scalar mn2=r(mean)
	eret2 scalar sd2=r(sd)
	est store `var'
}
estout * using "table7.txt", drop($area_controls _cons) title("Table 7: Social effects, Endline 2") ///
	prehead("" @title) cells(b(fmt(a3) s) se(fmt(a3) par(`"="("'`")""'))) append s(r2 mn2 sd2 N pval) ///
	starlevels(* .1 ** .05 *** .01) legend ///
	postfoot("Robust standard errors, clustered at the area level, in brackets.")

	
********************************************************************************
********             Table: Adjusted P-values for Indices              *********
********************************************************************************
use "$datadir/2013-0533_data_endlines1and2.dta", clear

*ENDLINE 1 INDICES:
local EL1_indices "credit_index biz_index_all biz_index_old biz_index_new income_index labor_index total_exp_mo_pc social_index"
local EL1_tablenos "2 3 3B 3C 4 5 6 7"
local EL2_indices "credit_index biz_index_all biz_index_old income_index labor_index total_exp_mo_pc social_index"
local EL2_tablenos "2 3 3B 4 5 6 7"

forval i = 1/2 {
	preserve
	
	gen str tableno=""
	gen str index=""
	gen pval_unadj_`i'=.
	
	local j 1
	foreach var in `EL`i'_indices'   {
		replace index="`var'" if _n==`j'
		reg `var'_`i' treatment $area_controls [pweight=w`i'], cluster(areaid)
		replace pval_unadj_`i'=2*ttail(e(df_r),abs(_b[treatment]/_se[treatment])) if _n==`j'
		local tableno: word `j' of `EL`i'_tablenos'
		replace tableno="`tableno'" if _n==`j'
		local j = `j'+1
		}
	keep tableno index pval_unadj_`i'
	keep if index!=""
	
	*Compute adjusted p-values using Hochberg's step-up method:
	gsort -pval_unadj_`i'
	gen pval_Hochberg_`i' = pval_unadj_`i' * (_N + 1 - _n)
	
	tempfile el`i'_pvals
	save `el`i'_pvals'
	restore
}

use `el1_pvals', clear
merge 1:1 tableno index using `el2_pvals', assert(master matched) nogen
outsheet using "table_index_pvals.txt", replace


********************************************************************************
****      Table A1: Treatment-Control Balance in Fixed Characteristics     *****
********************************************************************************
use "$datadir/2013-0533_data_endlines1and2.dta", clear

*PANEL A: ENDLINE 1
est clear
scalar drop _all
foreach var in spouse_literate_1 spouse_works_wage_1 hhsize_1 women1845_1 anychild1318_1 ///
			   old_biz ownland_hyderabad_1 ownland_village_1 {
	reg `var' treatment if sample1==1 [pweight=w1], cluster(areaid)
	sum `var' if treatment==0 & e(sample)
	eret2 scalar mn1=r(mean)
	eret2 scalar sd1=r(sd)
	est store `var'
}
estout * using "tableA1.txt", drop(_cons) title("Table A1: Balance in fixed characteristics, Endline 1") ///
	prehead("" @title) cells(b(fmt(a3) s) se(fmt(a3) par(`"="("'`")""'))) replace s(mn1 sd1 N) ///
	starlevels(* .1 ** .05 *** .01) legend ///
	postfoot("Robust standard errors, clustered at the area level, in brackets.")

reg treatment spouse_literate_1 spouse_works_wage_1 hhsize_1 women1845_1 anychild1318_1 ///
			   old_biz ownland_hyderabad_1 ownland_village_1 [pweight=w1], cluster(areaid)
scalar Fstat = e(F)
scalar Fprob = Ftail(e(df_m), e(df_r), e(F))
matrix TableA1_A = [Fstat \ Fprob]
matrix rownames TableA1_A = F_stat F_pval
mat2txt, matrix(TableA1_A) saving("tableA1.txt") append
	
*PANEL B: ENDLINE 2
est clear
scalar drop _all
foreach var in spouse_literate_2 spouse_works_wage_2 hhsize_2 women1845_2 anychild1318_2 ///
			   old_biz ownland_hyderabad_2 ownland_village_2 {
	reg `var' treatment if sample2==1 [pweight=w2], cluster(areaid)
	sum `var' if treatment==0 & e(sample)
	eret2 scalar mn2=r(mean)
	eret2 scalar sd2=r(sd)
	est store `var'
}
estout * using "tableA1.txt", drop(_cons) title("Table A1: Balance in fixed characteristics, Endline 2") ///
	prehead("" @title) cells(b(fmt(a3) s) se(fmt(a3) par(`"="("'`")""'))) append s(mn2 sd2 N) ///
	starlevels(* .1 ** .05 *** .01) legend ///
	postfoot("Robust standard errors, clustered at the area level, in brackets.")

reg treatment spouse_literate_2 spouse_works_wage_2 hhsize_2 women1845_2 anychild1318_2 ///
			   old_biz ownland_hyderabad_2 ownland_village_2 [pweight=w2], cluster(areaid)
scalar Fstat = e(F)
scalar Fprob = Ftail(e(df_m), e(df_r), e(F))
matrix TableA1_B = [Fstat \ Fprob]
matrix rownames TableA1_B = F_stat F_pval
mat2txt, matrix(TableA1_B) saving("tableA1.txt") append


********************************************************************************
*********                Table A2: Endline 1 Attrition                **********
********************************************************************************
use "$datadir/2013-0533_data_census", clear

*NOTE: Attrition data for areas 61 and 62 was unreliable; these areas are omitted
*	   from the following analysis (the "attrit" indicator is set to missing for
*	   all households in these two survey areas).

*PANEL A: EL1 ATTRITION IN TREATMENT VS. CONTROL
scalar drop _all
gen found_EL1 = 1-attrit
sum found_EL1 if treatment==1
scalar treat_mean=r(mean)	
sum found_EL1 if treatment==0
scalar control_mean=r(mean)
reg found_EL1 treatment, cluster(areaid)
test treatment=0
scalar pval=r(p)
matrix TableA2 = [treat_mean \ control_mean \ pval]
matrix rownames TableA2 = mean_treatment mean_control diff_pval
mat2txt, matrix(TableA2) saving("tableA2.txt") replace

*PANEL B: EL1 ATTRITION BY HOUSEHOLD CHARACTERISTICS (CENSUS)
est clear
local reg1 "treatment"
local reg2 "treatment spandana_borrower pucca hhinslum_months woman_biz woman_salary husb_biz husb_salary"
local reg3 "firstloandate"
local reg4 "p10loandate"

forval i=1/4 {
	reg attrit `reg`i'', cluster(areaid)
	est sto EL1attrit`i'
}
 
estout EL1attrit* using "tableA2.txt", title("Table A2: Endline 1 attrition") ///
	prehead("" @title) cells(b(fmt(a3) s) se(fmt(a3) par(`"="("'`")""'))) append s(r2 N) ///
	starlevels(* .1 ** .05 *** .01) legend ///
	postfoot("Robust standard errors, clustered at the area level, in brackets.")


********************************************************************************
*********                Table A3: Endline 2 Attrition                **********
********************************************************************************
use "$datadir/2013-0533_data_endlines1and2.dta", clear

*Generate indicator for "attrited between endline 1 and endline 2":
gen attrit_el2 = 1 - sample2

*Rescale monetary variables to thousands of 2007 Rs.:
foreach var of varlist total_exp_mo_pc_1 temptation_exp_mo_pc_1 durables_exp_mo_pc_1 ///
					   festival_exp_mo_pc_1 bizprofit_1 {
	replace `var'=`var'/1000
}

*PANEL A: EL2 ATTRITION IN TREATMENT VS. CONTROL
scalar drop _all
sum sample2 if treatment==1
scalar treat_mean=r(mean)	
sum sample2 if treatment==0
scalar control_mean=r(mean)
reg sample2 treatment, cluster(areaid)
test treatment=0
scalar pval=r(p)
matrix TableA3 = [treat_mean \ control_mean \ pval]
matrix rownames TableA3 = mean_treatment mean_control diff_pval
mat2txt, matrix(TableA3) saving("tableA3.txt") replace

*PANEL B: EL2 ATTRITION BY HOUSEHOLD CHARACTERISTICS (ENDLINE 1)
est clear
foreach var of varlist total_exp_mo_pc_1 temptation_exp_mo_pc_1 durables_exp_mo_pc_1 festival_exp_mo_pc_1 ///
					   spandana_1 anymfi_1 newbiz_1 old_biz bizprofit_1 {
	preserve
	rename `var' EL1_characteristic
	reg attrit_el2 EL1_characteristic [pweight=w1], cluster(areaid)
	est sto `var'
	restore
}
estout * using "tableA3.txt", title("Table A3: Endline 2 attrition by household characteristics") ///
	prehead("" @title) cells(b(fmt(a3) s) se(fmt(a3) par(`"="("'`")""'))) append s(r2 N) ///
	starlevels(* .1 ** .05 *** .01) legend ///
	postfoot("Robust standard errors, clustered at the area level, in brackets.")

*PANEL B: EL2 ATTRITION BY HOUSEHOLD CHARACTERISTICS, TREATMENT VS. CONTROL (ENDLINE 1)
est clear
foreach var of varlist total_exp_mo_pc_1 temptation_exp_mo_pc_1 durables_exp_mo_pc_1 festival_exp_mo_pc_1 ///
					   spandana_1 anymfi_1 newbiz_1 old_biz bizprofit_1 {
	preserve
	rename `var' EL1_characteristic
	gen treat_EL1_characteristic = treatment * EL1_characteristic
	reg attrit_el2 EL1_characteristic treatment treat_EL1_characteristic [pweight=w1], cluster(areaid)
	est sto `var'
	restore
}
estout * using "tableA3.txt", drop(EL1_characteristic treatment) ///
	title("Table A3: Endline 2 attrition by household characteristics, treatment vs. control") ///
	prehead("" @title) cells(b(fmt(a3) s) se(fmt(a3) par(`"="("'`")""'))) append s(r2 N) ///
	starlevels(* .1 ** .05 *** .01) legend ///
	postfoot("Robust standard errors, clustered at the area level, in brackets.")
	
	
********************************************************************************
*********        Table A4: Industries of Businesses, Endline 1        **********
********************************************************************************
use "$datadir/2013-0533_data_endline1businesstype.dta", clear
merge m:1 hhid using "$datadir/2013-0533_data_endlines1and2.dta", keepusing(w1) keep(matched) assert(using matched) nogen

*1) Generate indicators for each industry:
local i=1
foreach x in clothing food_agr repair_construction crafts_vendor rickshaw_driving other {
	gen `x'=(business_type_aggregate_1==`i') if !mi(business_type_aggregate_1)
	local i=`i'+1
}

*2) Create summary statistics table:
scalar drop _all
foreach var of varlist food_agr clothing rickshaw_driving repair_construction crafts_vendor other {
	*Old businesses:
	sum `var' if treatment==1 & new_business_1==0
	scalar `var'_old_t=r(mean)	
	sum `var' if treatment==0 & new_business_1==0
	scalar `var'_old_c=r(mean)
	
	reg `var' treatment if new_business_1==0 [pweight=w1], cluster(areaid)
	scalar `var'_old_d=_b[treatment]
	scalar `var'_old_se=_se[treatment]
	test treatment=0
	scalar `var'_old_pval=r(p)
	
	*New businesses:
	sum `var' if treatment==1 & new_business_1==1
	scalar `var'_new_t=r(mean)	
	sum `var' if treatment==0 & new_business_1==1
	scalar `var'_new_c=r(mean)
	
	reg `var' treatment if new_business_1==1 [pweight=w1], cluster(areaid)
	scalar `var'_new_d=_b[treatment]
	scalar `var'_new_se=_se[treatment]
	test treatment=0
	scalar `var'_new_pval=r(p)	
	
	matrix TableA4 = [nullmat(TableA4) \ `var'_old_t, `var'_old_c, `var'_old_d, `var'_old_se, `var'_old_pval, ///
										 `var'_new_t, `var'_new_c, `var'_new_d, `var'_new_se, `var'_new_pval]
}

sum business_type_aggregate_1 if treatment==1 & new_business_1==0
scalar Old_treat_nobs=r(N) 
sum business_type_aggregate_1 if treatment==0 & new_business_1==0
scalar Old_control_nobs=r(N)
sum business_type_aggregate_1 if treatment==1 & new_business_1==1
scalar New_treat_nobs=r(N) 
sum business_type_aggregate_1 if treatment==0 & new_business_1==1
scalar New_control_nobs=r(N)
matrix TableA4 = [nullmat(TableA4) \ Old_treat_nobs, Old_control_nobs, ., ., ., ///
									 New_treat_nobs, New_control_nobs, ., ., .]

matrix rownames TableA4 = food_agr clothing rickshaw_driving repair_construction crafts_vendor other nobs
matrix colnames TableA4 = Old_treat_mean Old_control_mean Old_difference Old_se Old_p_val ///
						  New_treat_mean New_control_mean New_difference New_se New_p_val
mat2txt, matrix(TableA4) saving("tableA4.txt") replace


********************************************************************************
*********            Table A5: Attrition-Corrected Results            **********
********************************************************************************
use "$datadir/2013-0533_data_endlines1and2.dta", clear

*Predict propensity of being observed at endline 2 as function of observables:
local observables ""
foreach var of varlist total_exp_mo_pc_1 temptation_exp_mo_pc_1 durables_exp_mo_pc_1 ///
					   anymfi_1 anymfi_amt_1 newbiz_1 old_biz bizprofit_1  {
	gen `var'_m   = (`var'==.)
	gen `var'_reg = `var'
	replace `var'_reg=0 if `var'==.
	gen `var'_treat   = `var'_reg*treat
	gen `var'_m_treat =	`var'_m*treat
	local observables "`observables' `var'_reg `var'_treat `var'_m `var'_m_treat"
}
probit sample2 `observables' [pweight=w1], cluster(areaid)
predict sample_hat

*Reweight regressions by inverse of predicted propensity of being observed:
gen w1_attrit = w1 * (1/sample_hat)
gen w2_attrit = w2 * (1/sample_hat)

*PANEL A: ENDLINE 1
est clear
foreach var in total_exp_mo_pc_1 temptation_exp_mo_pc_1 durables_exp_mo_pc_1 anymfi_1 newbiz_1  ///
			   female_biz_new_1 bizprofit_1  hours_headspouse_biz_1 women_emp_index_1 {
	reg `var' treatment $area_controls [pweight=w1_attrit], cluster(areaid)
	eret2 scalar pval=2*ttail(e(df_r),abs(_b[treatment]/_se[treatment]))
	sum `var' if treatment==0 & e(sample)
	eret2 scalar mn1=r(mean)
	eret2 scalar sd1=r(sd)
	est store `var'
}
estout * using "tableA5.txt", drop($area_controls _cons) title("Table A5: Attrition-corrected results, Endline 1") ///
	prehead("" @title) cells(b(fmt(a3) s) se(fmt(a3) par(`"="("'`")""'))) replace s(r2 mn1 sd1 N pval) ///
	starlevels(* .1 ** .05 *** .01) legend ///
	postfoot("Robust standard errors, clustered at the area level, in brackets.")

*PANEL B: ENDLINE 2
est clear
foreach var in total_exp_mo_pc_2 temptation_exp_mo_pc_2 durables_exp_mo_pc_2 anymfi_2 newbiz_2  ///
			   female_biz_new_2 bizprofit_2  hours_headspouse_biz_2 women_emp_index_2 {
	reg `var' treatment $area_controls [pweight=w2_attrit], cluster(areaid)
	eret2 scalar pval=2*ttail(e(df_r),abs(_b[treatment]/_se[treatment]))
	sum `var' if treatment==0 & e(sample)
	eret2 scalar mn1=r(mean)
	eret2 scalar sd1=r(sd)
	est store `var'
}
estout * using "tableA5.txt", drop($area_controls _cons) title("Table A5: Attrition-corrected results, Endline 2") ///
	prehead("" @title) cells(b(fmt(a3) s) se(fmt(a3) par(`"="("'`")""'))) append s(r2 mn1 sd1 N pval) ///
	starlevels(* .1 ** .05 *** .01) legend ///
	postfoot("Robust standard errors, clustered at the area level, in brackets.")
	
	
********************************************************************************
****     Table 6: Add'l self-employment outcomes (all households)          *****
********************************************************************************
use "$datadir/2013-0533_data_endlines1and2.dta", clear

*PANEL A: ENDLINE 1
est clear
foreach var in bizrev_1	any_new_biz_1 female_biz_new_1 {
	reg `var' treatment $area_controls [pweight=w1], cluster(areaid)
	eret2 scalar pval=2*ttail(e(df_r),abs(_b[treatment]/_se[treatment]))
	sum `var' if treatment==0 & e(sample)
	eret2 scalar mn1=r(mean)
	eret2 scalar sd1=r(sd)
	est store `var'
}
estout * using "table3.txt", drop($area_controls _cons) title("Table 3: Self-employment activities, Endline 1") ///
	prehead("" @title) cells(b(fmt(a3) s) se(fmt(a3) par(`"="("'`")""'))) replace s(r2 mn1 sd1 N pval) ///
	starlevels(* .1 ** .05 *** .01) legend ///
	postfoot("Robust standard errors, clustered at the area level, in brackets.")	

*PANEL B: ENDLINE 2
est clear
foreach var in bizrev_2	any_new_biz_2 female_biz_new_2 {
	reg `var' treatment $area_controls [pweight=w2], cluster(areaid)
	eret2 scalar pval=2*ttail(e(df_r),abs(_b[treatment]/_se[treatment]))
	sum `var' if treatment==0 & e(sample)
	eret2 scalar mn2=r(mean)
	eret2 scalar sd2=r(sd)
	est store `var'
}
estout * using "table3.txt", drop($area_controls _cons) title("Table 3: Self-employment activities, Endline 2") ///
	prehead("" @title) cells(b(fmt(a3) s) se(fmt(a3) par(`"="("'`")""'))) append s(r2 mn2 sd2 N pval) ///
	starlevels(* .1 ** .05 *** .01) legend ///
	postfoot("Robust standard errors, clustered at the area level, in brackets.")

	
log close
