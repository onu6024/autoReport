# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 14:35:04 2023

@author: ninewatt
"""

# 분석 라이브러리 읽어오기 
import os
import pandas as pd
import numpy as np
import psycopg2 as pg
import pandas.io.sql as psql
from datetime import timedelta
import datetime as dt
import math
import OPFA_fun as fun
from statsmodels.tsa.seasonal import seasonal_decompose
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from yellowbrick.cluster.elbow import kelbow_visualizer
from copy import deepcopy
from scipy import stats
from CPM_function import PiecewiseRegression
import powerPlanner as pp
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import july
from july.utils import date_range
from matplotlib.patches import Rectangle
import bisect
from pytimekr import pytimekr
import calendar
from db_connect import DBConnection

#DB conn & query
db = DBConnection()
db.connect()

#### 1_고객정보 
def custInfo(custNo, custNm, start, end):
    #DB conn & query
    db = DBConnection()
    db.connect()

    plt.rcParams["font.family"]='Malgun Gothic'
    plt.rcParams['axes.unicode_minus']=False
    
    #계약정보 불러오기 - 고정값 
    sql="""SELECT * FROM ereport.sei_custinfo WHERE "custNo" = '{}'""".format(custNo)
    custinfo=db.execute_query(sql)

    #고객주소 정보 불러오기 
    # sql="""SELECT * FROM ereport.sei_address WHERE "custNo" = '{}'""".format(custNo)
    # address=db.execute_query(sql)

    #요금청구정보, 전력사용량 불러오기 - 검침일 기준 
    sql="""SELECT * FROM ereport.sei_bill WHERE "custNo"= '{}' AND "billYm" BETWEEN '{}' AND '{}'""".format(custNo, start[0:7], end[0:7])
    billdata=db.execute_query(sql)

    #15분 단위 전력 데이터 불러오기 - minute : 결측치 처리 안 되어 있음, 다계기 없음(과거데이터)
    sql="""SELECT * FROM ereport.sei_usekwh WHERE "custNo"= '{}' AND "mrYmd" BETWEEN '{}' AND '{}'""".format(custNo, start, end)
    minute=db.execute_query(sql, dtype={'mrYmd':'datetime64[ns]'})
    minute.set_index("mrYmd", inplace=True)
    
    #결측치 0처리 
    r=pd.date_range(start=start, end=pd.to_datetime(end)+dt.timedelta(0,-900,0), freq='15min', name='mrYmd')
    data_minute=pd.DataFrame(index=r)
    data_minute=data_minute.join(minute)
    data_minute.fillna(0, inplace=True)

    #15분 -> 1시간으로 가공하기 
    hourly=pd.DataFrame()
    hourly["usekWh"]=data_minute.resample('1H').sum()["pwrQty"]
    hourly["demandkW"]=data_minute.resample('1H').max()["pwrQty"]*4
    hourly["demandMean"]=hourly["demandkW"].mean()
    
    #1시간 -> 1일로 가공하기 
    daily=pd.DataFrame()
    daily["usekWh"]=hourly["usekWh"].resample('1D').sum()
    daily["demand"]=hourly["demandkW"].resample('1D').max()
    
    #분석으로 인한 절감 가능성 체크 
    if custinfo["cntrPwr"][0]>1000:    
        ikwStatus=0
    else:
        ikwStatus=1
    
    #### Return ####
    front=pd.DataFrame({'custNm' : [custNm],  #고객명
                        'anlyPrdStart': [start], #분석 시작일
                        'anlyPrdEnd' : [end],    #분석 종료일,
                        'reportDate' : [datetime.now().strftime('%Y-%m-%d')],   #리포트 생성일
                        })
    
    #고객정보일반 
    custInfo=pd.DataFrame({'custNo':[custNo],   #고객번호
                           'cntrPwr': [custinfo["cntrPwr"][0].astype(int)], #계약전력
                           'cntrKnd':[custinfo["cntrKnd"][0]],  #계약종별
                           'selCost':[custinfo['selCost'][0]],  #선택요금제
                           'pchildC':[custinfo['pchildClcd'][0]],  #모자구분
                           'usekwh': [round(billdata[["lloadkWh", "mloadkWh", "maxloadkWh"]].sum().sum(),1).astype('int64')],   #분석기간 내 전력사용량총합
                           'reqBill' : [billdata['reqBill'].sum().sum().astype('int64')], #분석기간 내 전기요금총합
                           'maxDemand' : [round(hourly["demandkW"].max(),1)], #분석기간 내 최대수요전력
                           'maxDemandTime': [hourly[hourly["demandkW"]==hourly["demandkW"].max().max()].iloc[0].name], #최대수요전력 시점
                           'ikwStatus': [ikwStatus] #계약전력 1000kW 초과 여부 
                           })
    
    
    #연간 전력 사용 패턴 분석 
    fig1=plt.figure(figsize=(16,3))
    plt.plot(hourly["demandkW"], label='시간별 최대수요')
    plt.plot(hourly["demandMean"], linestyle='--')
    plt.text(hourly.index[len(hourly)-1].date()+dt.timedelta(5,0,0), hourly["demandMean"][0]*0.95, "평균", color='tab:orange', fontsize=14)
    plt.grid(True)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel("수요전력(kW)", fontsize=14)
    current_values = plt.gca().get_yticks()
    plt.gca().set_yticklabels(['{:,.0f}'.format(x) for x in current_values])
    plt.title("연 평균 수요전력: {:,.0f}kW".format(round(hourly["demandMean"].iloc[0],1)), fontsize=16)
    
    plt.savefig("images\\custinfo_1.png", bbox_inches='tight',pad_inches=0.4)
    plt.close()
    
    #Anormaly : 일별 전력사용량 기준으로 이상가동일 추정 
    result=seasonal_decompose(daily["usekWh"], model='additive')
    constance=1.5
    
    Q1 = np.percentile(result.resid.dropna(), 25)
    Q3 = np.percentile(result.resid.dropna(), 75)
    IQR = Q3 - Q1 
    IQR = IQR if IQR > 0 else -1*IQR
    lower = Q1 - constance*IQR
    higher = Q3 + constance*IQR
    
    #IQR 기준으로 이상치 판단하기 
    for i in result.resid.dropna().index:
        if result.resid.dropna().loc[i] < lower or  result.resid.dropna().loc[i] > higher:
            daily.at[i,"Anormaly"]=1
        else:
            daily.at[i,"Anormaly"]=0
            
    #결측값에 대해 이상치 판단하기 
    for i in daily[daily["Anormaly"].isna()].index:
        daily.at[i, "Anormaly"]=1
    
    holidayList=pytimekr.holidays(year=int(start[0:4]))+pytimekr.holidays(year=int(end[0:4]))
    holidayList=list(set(holidayList))

    daily['holiday']=''
    for i in daily.index:
        #공휴일인지 여부 판단하기 
        if daily.loc[i].name in holidayList:
            daily.at[i, 'holiday']='True'
        else:
            daily.at[i, 'holiday']='False'
        
        #Anormaly 중 공휴일 제외하기 
        if daily.loc[i]['holiday']=='True' and daily.loc[i]['Anormaly']==1:
            daily.at[i, "Anormaly"]==0
            daily.at[i, "usekWhAft"]=daily.at[i, "usekWh"]
        elif daily.loc[i]['holiday']=='Fasle' and daily.loc[i]['Anormaly']==1:
            daily.at[i, "usekWhAft"]=daily.usekWh.mean()
        else:
            daily.at[i, "usekWhAft"]=daily.at[i, "usekWh"]
        
    Anormaly=daily[daily["Anormaly"]==1][["usekWh"]].reset_index()
    Anormaly=Anormaly.rename(columns={'datetime' : 'date'})
    
    #anlyUsekWh : 가동중지 추정일 제외한 일별 사용량
    anlyUsekWh=daily[["usekWhAft"]].dropna().astype('int64').reset_index()
    # anlyUsekWh=anlyUsekWh.rename(columns={'datetime' : 'date'})
    
    
    #전력 사용량 캘린더
    dates=date_range(anlyUsekWh["mrYmd"][0], anlyUsekWh["mrYmd"][len(daily)-1])

    fig2=july.heatmap(dates=dates, 
                         data=anlyUsekWh.usekWhAft, 
                         cmap='Blues',
                         month_grid=True, 
                         horizontal=True,
                         value_label=False,
                         date_label=False,
                         weekday_label=True,
                         month_label=True, 
                         year_label=True,
                         colorbar=True,
                         fontfamily="monospace",
                         fontsize=12,
                         title=None,
                         titlesize='large')
    
    plt.savefig("images\\custinfo_2.png", bbox_inches='tight',pad_inches=0.4)
    plt.close()
    
    #UsekWhStat : 요일별/월별 통계 
    #0:월, 1:화, 2:수, 3:목, 4:금, 5:토, 6:일
    data_day=pd.DataFrame(hourly["usekWh"].groupby(hourly.index.weekday).sum()).sort_values(by='usekWh', ascending=False)
    data_month=pd.DataFrame(hourly["usekWh"].groupby(hourly.index.month).sum()).sort_values(by='usekWh', ascending=False)
    
    usekWhStats=pd.DataFrame({'FirstMonth' : [data_month.iloc[0].name],
                             'SecondMonth' : [data_month.iloc[1].name],
                             'LastMonth' : [data_month.iloc[len(data_month)-1].name],
                             'FirstDay' : [data_day.iloc[0].name],
                             'SecondDay' : [data_day.iloc[1].name],
                             'LastDay' : [data_day.iloc[len(data_day)-1].name]}).astype('int64')
    
    #분석기간 중 마지막 날 00:00 제외하기
    dataClust=hourly['usekWh'].copy()
    
    #timeseries data k-means
    ts_value=dataClust.values.reshape(int(len(dataClust)/24),24)
    ts_value=np.nan_to_num(ts_value)
    scaler=TimeSeriesScalerMeanVariance(mu=0, std=1)
    data_scaled=scaler.fit_transform(ts_value)
    data_scaled=np.nan_to_num(data_scaled)

    #Euclidean k-means 모델 만들기 
    plt.clf()
    km=TimeSeriesKMeans(random_state=0)
    visualizer = kelbow_visualizer(km,ts_value, k=(2,9), show=False)
    number=visualizer.elbow_value_
    km=TimeSeriesKMeans(n_clusters=number, metric="euclidean", verbose=False, random_state=0)
    y_predicted = km.fit_predict(data_scaled)

    # resultClust : 일별 클러스터링 결과 저장 
    resultClust=pd.DataFrame(ts_value.copy())
    resultClust=pd.concat([resultClust, pd.DataFrame(y_predicted, columns={'predicted'})], axis=1)

    centerClust=pd.DataFrame()
    countClust=pd.DataFrame()
    for yi in range(number):
        temp=resultClust[resultClust['predicted']==yi].drop(columns='predicted')
        tempMean=pd.DataFrame(round(temp.mean(),1), columns={'{}'.format(yi+1)})
        tempCount=pd.DataFrame({'{}'.format(yi+1) : [len(temp)]})
        
        centerClust=pd.concat([centerClust, tempMean], axis=1)
        countClust=pd.concat([countClust, tempCount], axis=1)
    
    #연간 대표 전력사용패턴 
    plt.rcParams["font.family"]='Malgun Gothic'
    plt.rcParams['axes.unicode_minus']=False    
    fig3=centerClust.plot(xlabel='시간', ylabel='전력사용량(kWh)')
    plt.savefig('images\\custinfo_3.png', bbox_inches='tight',pad_inches=0.4)    
    
    #파이그래프 그리기 
    pieClust=pd.DataFrame()
    for i in range(1, len(centerClust.columns)+1):
        temp=pd.DataFrame({'clust' : [i],
                           'sum' : [len(resultClust[resultClust['predicted']+1==i])]})
        pieClust=pd.concat([pieClust, temp], axis=0)                         
    
    fig4=plt.figure()
    plt.pie(pieClust['sum'], autopct='%.1f%%')
    plt.legend(pieClust['clust'])
    plt.tight_layout()
    plt.savefig('images\\custinfo_4.png', bbox_inches='tight',pad_inches=0.4)
    
    #증감 트렌드 도출하기
    centerDiff=centerClust.diff()
    centerGuide=pd.DataFrame()
    for i in centerDiff.columns:
        tempCenterDiff=centerDiff[i]
        # maxTime : 전력사용량 증가량 최대 시간 
        maxTime=tempCenterDiff[tempCenterDiff==tempCenterDiff.max()].index[0]
        Temp=pd.DataFrame({'{}'.format(i):[maxTime]}, index=['maxTime'])
        
        centerGuide=pd.concat([centerGuide, Temp], axis=1)
    
    return custInfo, Anormaly, usekWhStats, fig1, fig2, fig3, fig4, centerGuide


#### 2_선택요금제 변경

def selCost(custNo, custNm, start, end):
    #DB conn & query
    db = DBConnection()
    db.connect()

    plt.rcParams["font.family"]='Malgun Gothic'
    plt.rcParams['axes.unicode_minus']=False    
    
    #전력요금단가 불러오기  - sei_cost
    sql="""SELECT * FROM ereport.sei_cost"""
    info_costn=db.execute_query(sql)

    #월별계절정보
    info_month=pd.DataFrame({'month':['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'], 
                             'season':['winter', 'winter', 'sf', 'sf', 'sf', 'summer', 'summer', 'summer', 'sf', 'sf', 'winter', 'winter']})
    
    #고객의 현재 요금제 정보 불러오기 
    sql="""SELECT * FROM ereport.sei_custinfo WHERE "custNo"='{}'""".format(custNo)
    custInfo=db.execute_query(sql)
    item=custInfo["cntrKnd"][0]    
    selCost=custInfo['selCost'][0]

    infoFare=info_costn[info_costn["item"]==item].reset_index(drop=True)
    infoFare=infoFare.drop(columns=['id','editDate','envCost','fuelCost','created_at'])
    if len(info_costn[info_costn["item"]==item]["sel"].unique())>1:
        infoFare=infoFare[infoFare["sel"]==selCost]    

    #선택요금제 분석이 가능한지 여부 확인 - 저압 등 선택요금 선택이 불가한 경우는 분석에서 제외
    if len(info_costn[info_costn["item"]==item]["sel"].unique())>1:
        
        item_cost=info_costn[info_costn["item"]==item]
        
        #부하별 사용량 데이터 불러오기 - sei_bill table에서 읽어오기 
        sql="""SELECT * FROM ereport.sei_bill WHERE "custNo"= '{}' AND "billYm" BETWEEN '{}' AND '{}'""".format(custNo, start[0:7], end[0:7])
        detail_charge=db.execute_query(sql)
        detail_charge=detail_charge.reset_index(drop=True)
        # detail_charge=detail_charge.astype({'billAplyPwr':'int', 'lload_usekWh':'int', 'mload_usekWh':'int', 'maxload_usekWh':'int', 'baseBill':'int', 'kwhBill':'int', 'reqBill':'int'})
        detail_charge["total_usekWh"]=detail_charge[["lloadkWh","mloadkWh","maxloadkWh"]].sum(axis=1)
        
        for i in range(0, len(detail_charge)):
            detail_charge.at[i, 'date']=detail_charge.iloc[i]['billYm'][0:4]+detail_charge.iloc[i]['billYm'][5:7]
            detail_charge.at[i, 'month']=detail_charge.iloc[i]['billYm'][5:7]
        
        detail_charge.index=detail_charge['date']
        detail_charge=detail_charge.sort_index()
        detail_charge['avgReqBill']=round(detail_charge["reqBill"].mean(),1)
        
        #anlySelGraph1 
        anlySelGraph1=detail_charge[["date", "avgReqBill", "lloadkWh", "mloadkWh", "maxloadkWh", "reqBill"]].astype(int)
        
        fig1=plt.figure(figsize=(16,3))
        fig1.set_facecolor('white')

        #회색(경부하):7f7f7f / 초록색(중간부하):#5C8FE6 / 붉은색(최대부하):#B32712
        ax1=fig1.add_subplot()
        ax1.bar(anlySelGraph1["date"].astype(str), anlySelGraph1["lloadkWh"], width=0.4, color='lightgray', label='경부하')
        ax1.bar(anlySelGraph1["date"].astype(str), anlySelGraph1["mloadkWh"], bottom=anlySelGraph1["lloadkWh"], width=0.4, color='#5C8FE6', label='중간부하')
        ax1.bar(anlySelGraph1["date"].astype(str), anlySelGraph1["maxloadkWh"], bottom=np.add(anlySelGraph1["lloadkWh"], anlySelGraph1["mloadkWh"]), width=0.4, color='#B32712', label='최대부하')
        current_values = ax1.get_yticks()
        ax1.set_yticklabels(['{:,.0f}'.format(x) for x in current_values])
        plt.ylabel("전력사용량(kWh)", fontsize=14)
        plt.grid(True)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(bbox_to_anchor=(0.7,-0.12), ncol=3, fontsize=14)

        ax2=ax1.twinx()
        ax2.plot(anlySelGraph1["date"].astype(str), anlySelGraph1["reqBill"], linestyle='--', marker='o', linewidth=3, color='#007380')
        ax2.tick_params(axis='y')
        current_values = ax2.get_yticks()
        ax2.set_yticklabels(['{:,.0f}'.format(x) for x in current_values/(10**4)])
        plt.ylabel("만원", fontsize=14, color='#007380')
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14, color='#1f77b4')
        fig1.suptitle("월 평균 전기요금 : {}만원".format(round(anlySelGraph1["reqBill"].mean()/(10**4), 1)), fontsize=16)

        # fig1.tight_layout()
        plt.savefig("images\\selcost_1.png", bbox_inches='tight',pad_inches=0.4)
        
        
        
        #선택요금제 비교
        detail_charge=pd.merge(detail_charge, info_month, how='outer', on='month')
        detail_charge=detail_charge.dropna(subset=['reqBill'])
        
        cyber_kepco_cost=pd.DataFrame(columns={"baseBill", "kWhBill", "envBill", "fuelBill"}, index=item_cost["sel"].unique())
                
        
        #부하별 요금제가 아닌 경우 - 저압 
        if item_cost["loadNm"].iloc[0]=='nan':
            for i in item_cost["sel"].unique():
                temp=detail_charge[["month", "billAplyPwr", "lloadkWh", "mloadkWh", "maxloadkWh", "season"]].astype(dtype=int, errors='ignore')
                temp=temp.dropna()
                temp["baseBill"]=temp["billAplyPwr"]*item_cost[item_cost["sel"]==i]["basic"].iloc[0]
                for j in range(0, len(temp)):
                    temp.at[j, "kWhBill"]=temp.iloc[j]["mloadkWh"]*item_cost[item_cost["sel"]==i][temp.iloc[j]["season"]].iloc[0]
                    temp.at[j, "envBill"]=temp.iloc[j][["lloadkWh", "mloadkWh", "maxloadkWh"]].sum()*9 #기후환경요금단가 : 9원/kWh
                    temp.at[j, "fuelBill"]=temp.iloc[j][["lloadkWh", "mloadkWh", "maxloadkWh"]].sum()*5 #연료비조정단가 : 5원/kWh
                    temp.at[j, "요금합"]=temp.iloc[j][["baseBill", "kWhBill", "envBill", "fuelBill"]].sum()
                    temp.at[j, "부가가치세"]=round(temp.iloc[j]["요금합"]*0.1)
                    temp.at[j, "전력사업기반기금"]=math.trunc(temp.iloc[j]["요금합"]*0.037/10)*10
                    
                    cyber_kepco_cost.at[i, "baseBill"]=temp["baseBill"].sum()
                    cyber_kepco_cost.at[i, "kWhBill"]=temp["kWhBill"].sum()
                    cyber_kepco_cost.at[i, "envBill"]=temp["envBill"].sum()
                    cyber_kepco_cost.at[i, "fuelBill"]=temp["fuelBill"].sum()
                    cyber_kepco_cost.at[i, "요금합"]=temp["요금합"].sum()
                    cyber_kepco_cost.at[i, "부가가치세"]=temp["부가가치세"].sum()
                    cyber_kepco_cost.at[i, "전력사업기반기금"]=temp["전력사업기반기금"].sum()
        
        #그 외 요금제 
        else:
            for i in item_cost["sel"].unique():
                temp=detail_charge[["month", "billAplyPwr", "lloadkWh", "mloadkWh", "maxloadkWh", "season"]].astype(dtype=int, errors='ignore')
                temp=temp.dropna()
                temp["baseBill"]=temp["billAplyPwr"]*item_cost[item_cost["sel"]==i]["basic"].iloc[0]
                for j in range(0, len(temp)):
                    temp.at[j, "kWhBill"]=temp.iloc[j]["lloadkWh"]*item_cost[(item_cost["sel"]==i)&(item_cost["loadNm"]=='경부하')][temp.iloc[j]["season"]].iloc[0]+temp.iloc[j]["mloadkWh"]*item_cost[(item_cost["sel"]==i)&(item_cost["loadNm"]=='중간부하')][temp.iloc[j]["season"]].iloc[0]+temp.iloc[j]["maxloadkWh"]*item_cost[(item_cost["sel"]==i)&(item_cost["loadNm"]=='최대부하')][temp.iloc[j]["season"]].iloc[0]
                    temp.at[j, "envBill"]=temp.iloc[j][["lloadkWh", "mloadkWh", "maxloadkWh"]].sum()*9 #기후환경요금단가 : 9원/kWh
                    temp.at[j, "fuelBill"]=temp.iloc[j][["lloadkWh", "mloadkWh", "maxloadkWh"]].sum()*5 #연료비조정단가 : 5원/kWh
                    temp.at[j, "요금합"]=temp.iloc[j][["baseBill", "kWhBill", "envBill", "fuelBill"]].sum()
                    temp.at[j, "부가가치세"]=round(temp.iloc[j]["요금합"]*0.1)
                    temp.at[j, "전력사업기반기금"]=math.trunc(temp.iloc[j]["요금합"]*0.037/10)*10
                cyber_kepco_cost.at[i, "baseBill"]=temp["baseBill"].sum()
                cyber_kepco_cost.at[i, "kWhBill"]=temp["kWhBill"].sum()
                cyber_kepco_cost.at[i, "envBill"]=temp["envBill"].sum()
                cyber_kepco_cost.at[i, "fuelBill"]=temp["fuelBill"].sum()
                cyber_kepco_cost.at[i, "요금합"]=temp["요금합"].sum()
                cyber_kepco_cost.at[i, "부가가치세"]=temp["부가가치세"].sum()
                cyber_kepco_cost.at[i, "전력사업기반기금"]=temp["전력사업기반기금"].sum()
        
        cyber_kepco_cost=cyber_kepco_cost.astype(float)
        cyber_kepco_cost["totalBill"]=cyber_kepco_cost[["요금합", "부가가치세", "전력사업기반기금"]].sum(axis=1)
        
        cyber_kepco_cost["now_cost"]=''
        for i in cyber_kepco_cost.index:
            if i==selCost:
                cyber_kepco_cost.at[i, "now_cost"]=1
            else:
                cyber_kepco_cost.at[i, "now_cost"]=0
    
        #tableData
        tableData=cyber_kepco_cost[["baseBill", "kWhBill", "envBill", "fuelBill", "totalBill"]].astype('int64')
        tableData["sel"]=cyber_kepco_cost.index
        tableData=tableData.sort_values(by=['totalBill'], axis=0)
        
        pattern_last=detail_charge.iloc[len(detail_charge)-1]["total_usekWh"] #가장 최근 달의 총 전력사용량
        pattern_first=detail_charge.iloc[0]["total_usekWh"] #1년전 달의 총 전력사용량
    
        if pattern_first>pattern_last:
            trend="decrease"
        elif pattern_first==pattern_last:
            trend="maintain"
        else:
            trend="increase"
    
        savingCost_opt_cost=min(cyber_kepco_cost[cyber_kepco_cost["now_cost"]==1]["totalBill"])-min(cyber_kepco_cost["totalBill"])
    
        if savingCost_opt_cost==0:
            saving='0'
        elif savingCost_opt_cost >=200000:
            saving='up'
        else:
            saving='down'
    
        #코멘트1
        comment1=pd.DataFrame(columns=["now_plan", "opt_plan", "save_cost", "saving"])
        
        comment1.at[0, "now_plan"]=selCost #현재요금제
        comment1.at[0, "opt_plan"]=str(int(cyber_kepco_cost[cyber_kepco_cost["totalBill"]==min(cyber_kepco_cost["totalBill"])].iloc[0].name)) #최적요금제
        comment1.at[0, "save_cost"]=round(savingCost_opt_cost, 4)
        comment1.at[0, "saving"]=saving
        comment1.at[0, "trend"]=trend
  
        #anlySelGraph2
        detail_charge["useTime"]=round(detail_charge["total_usekWh"]/detail_charge["billAplyPwr"],1)  #월평균사용시간
        detail_charge["avgUseTime"]=round(detail_charge["useTime"].mean(),1)    #연평균사용시간
        
        anlySelGraph2=detail_charge[["avgUseTime", "date", "useTime"]]    

        fig2=plt.figure(figsize=(16,3))

        plt.bar(anlySelGraph2["date"], anlySelGraph2["useTime"], color='lightgrey', width=0.4, label='월간사용시간')
        plt.bar(anlySelGraph2[anlySelGraph2["useTime"]==anlySelGraph2["useTime"].max()]["date"].iloc[0], anlySelGraph2["useTime"].max(), width=0.4, color='#B32712')
        plt.bar(anlySelGraph2[anlySelGraph2["useTime"]==anlySelGraph2["useTime"].min()]["date"].iloc[0], anlySelGraph2["useTime"].min(), width=0.4, color='#124EB3')
        plt.plot(anlySelGraph2["date"], anlySelGraph2["avgUseTime"], linestyle='--', label='연평균 사용시간', color='#007380')
        plt.ylabel("시간", fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        fig2.suptitle("연간 월 평균 전력 사용시간 : {} 시간".format(round(anlySelGraph2["avgUseTime"][0], 1)), fontsize=16)
        plt.text(11.3, anlySelGraph2["avgUseTime"].mean().astype(float)*0.95, '평균', fontsize=12, color='#007380')
        # plt.legend(bbox_to_anchor=(0.65,-0.12), ncol=2, fontsize=14)
        plt.grid(True)

        plt.savefig("images\\selcost_2.png", bbox_inches='tight',pad_inches=0.4)
        
        
        
        #코멘트2
        comment2=pd.DataFrame(columns=["avgUseTime", "maxTime", "maxMonth", "minTime", "minMonth"])
        
        comment2.at[0, "avgUseTime"]=round(detail_charge["avgUseTime"][0], 1)
        comment2.at[0, "maxTime"]=round(detail_charge["useTime"].max(), 1)
        comment2.at[0, "maxMonth"]=detail_charge[detail_charge["useTime"]==detail_charge["useTime"].max()]["date"].iloc[0][4:7]
        comment2.at[0, "minTime"]=round(detail_charge["useTime"].min(), 1)
        comment2.at[0, "minMonth"]=detail_charge[detail_charge["useTime"]==detail_charge["useTime"].min()]["date"].iloc[0][4:7]
        
        selFare=item_cost[item_cost["sel"]==selCost].reset_index(drop=True)
        selFare=selFare.drop(columns=['id'])    
    else:
        
        anlySelGraph1=0
        tableData=pd.DataFrame()
        comment1=0 
        anlySelGraph2=0 
        comment2=0 
        selFare=0 
        fig1=0 
        fig2=0
        print("선택요금이 적용되는 계약종별이 아닙니다.")
        
    return infoFare, anlySelGraph1, tableData, comment1, anlySelGraph2, comment2, selFare, fig1, fig2


#### 3&4_계약전력변경&피크최적화
def anlyCtrPeak(custNo, custNm, start, end, GoalPeak, GoalTime):
    #DB conn & query
    db = DBConnection()
    db.connect()

    plt.rcParams["font.family"]='Malgun Gothic'
    plt.rcParams['axes.unicode_minus']=False    

    #전력단가 불러오기 
    sql="""SELECT * FROM ereport.sei_cost"""
    info_costn=db.execute_query(sql)

    #15분 단위 전력 데이터 불러오기 - minute : 결측치 처리 안 되어 있음, 다계기 없음(과거데이터)
    sql="""SELECT * FROM ereport.sei_usekwh WHERE "custNo"= '{}' AND "mrYmd" BETWEEN '{}' AND '{}'""".format(custNo, start, end)
    minute=db.execute_query(sql, dtype={'mrYmd':'datetime64[ns]'})
    minute.set_index("mrYmd", inplace=True)
    
    #결측치 0처리 
    r=pd.date_range(start=start, end=pd.to_datetime(end)+dt.timedelta(0,-900,0), freq='15min', name='mrYmd')
    data_minute=pd.DataFrame(index=r)
    data_minute=data_minute.join(minute)
    data_minute.fillna(0, inplace=True)
 
    #15분 -> 1시간으로 가공하기 
    hourly=pd.DataFrame()
    hourly["usekWh"]=data_minute.resample('1H').sum()["pwrQty"]
    hourly["demandkW"]=data_minute.resample('1H').max()["pwrQty"]*4
    hourly["demandMean"]=hourly["demandkW"].mean()
    
    #계약정보 불러오기 
    sql="""SELECT * FROM ereport.sei_custinfo WHERE "custNo"='{}'""".format(custNo)
    custInfo=db.execute_query(sql)
    cpower=custInfo['cntrPwr'][0]
    cpower30=round(cpower*0.3, 0)
    
    if info_costn["sel"][0]=="0" or 'nan':
        BaseCost=info_costn[info_costn["item"]==custInfo["cntrKnd"][0]].iloc[0]["basic"]  #기본요금
    else:     
        BaseCost=info_costn[(info_costn["item"]==custInfo["cntrKnd"][0])&(info_costn["sel"]==custInfo["selCost"][0])].iloc[0]["basic"]  #기본요금
    
    #청구요금 정보 가져오기  
    sql="""SELECT * FROM ereport.sei_bill WHERE "custNo"= '{}' AND "billYm" BETWEEN '{}' AND '{}'""".format(custNo, start[0:7], end[0:7])
    detail_charge=db.execute_query(sql)
    detail_charge=detail_charge.reset_index(drop=True) 
    
    for i in range(0, len(detail_charge)):
        detail_charge.at[i, 'date']=detail_charge.iloc[i]['billYm'][0:4]+detail_charge.iloc[i]['billYm'][5:7]
        detail_charge.at[i, 'month']=detail_charge.iloc[i]['billYm'][5:7]
    
    
    ####계약전력 변경####
    
    # 계약전력아 20kW 초과인 경우에만 요금적용전력이 피크 기반으로 결정됨 
    if custInfo["cntrPwr"][0]>=20:
        
        #anlyCntrTable
        hourly["month"]=hourly.index.strftime('%m')
        hourly["mr_ymd"]=hourly.index.date
        opt_cpower_table=hourly.groupby('month')['demandkW'].agg(**{'demandkW':'max'}).reset_index()
        opt_cpower_table=opt_cpower_table.rename(columns={"demandkW":"max_power"})
        opt_cpower_table["max_power"]=round(opt_cpower_table["max_power"])
        opt_cpower_table=pd.merge(opt_cpower_table, detail_charge[["billAplyPwr", "month", "date"]], on='month')
        
        anlyCntrTable=opt_cpower_table[["date", "max_power", "billAplyPwr"]]
        anlyCntrTable["demandkWPC"]=round(opt_cpower_table["max_power"]/custInfo["cntrPwr"][0]*100,1)
        anlyCntrTable=anlyCntrTable.sort_values('date').reset_index(drop=True)
        
        #comment
        opt_cpower_table["cpower"]=cpower
        opt_cpower_table["cpower30"]=cpower30
        opt_cpower_table["now_bill_aply_pwr"]=''
        
        opt_cpower_table=opt_cpower_table.sort_values('date')
        opt_cpower_table=opt_cpower_table.reset_index(drop=True)
        
        
        #월별 최대수요, 최소 요금적용전력 비교해서 현재 요금적용전력(now_bill_aply_pwr) 구하기
        for i in opt_cpower_table.index:
            if opt_cpower_table.at[i, "max_power"]>=opt_cpower_table["cpower30"][0]:
                opt_cpower_table.at[i, "now_bill_aply_pwr"]=opt_cpower_table.at[i, "max_power"]
            else:
                opt_cpower_table.at[i, "now_bill_aply_pwr"]=opt_cpower_table.at[i, "cpower30"]
    
        opt_cpower_table["now_bill_aply_pwr"]=opt_cpower_table["now_bill_aply_pwr"].astype(float)
    
        #월별 최대수요전력, 최소 요금적용전력 반영해 예상 요금적용전력 구하기 
        #이전 달의 최대수요전력, 최소 요금적용전력 고려해서 보수적으로 요금적용전력 산정하기 
        try:
            for i in opt_cpower_table.index:
                if opt_cpower_table.at[i, "now_bill_aply_pwr"]>=opt_cpower_table.at[i+1, "now_bill_aply_pwr"]:
                    opt_cpower_table.at[i+1, "now_bill_aply_pwr"]=opt_cpower_table.at[i, "now_bill_aply_pwr"]
                else:
                    opt_cpower_table.at[i+1, "now_bill_aply_pwr"]=opt_cpower_table.at[i+1, "now_bill_aply_pwr"]
    
        except KeyError:
            pass
    
    
        #opt_cpower : 최적의 계약전력 구하는 로직, 청구요금서 상의 최대수요전력 기반으로 계산, 공개불가 
        opt_cpower=''
        if min(opt_cpower_table["max_power"])>=cpower30:
            opt_cpower=cpower
        elif (min(opt_cpower_table["max_power"])<cpower30) & (max(opt_cpower_table["max_power"])*2>=round(min(opt_cpower_table["max_power"])/0.3)):
            opt_cpower=max(opt_cpower_table["max_power"])*2
        else:
            opt_cpower=round(min(opt_cpower_table["max_power"])/0.3)
            
        opt_cpower_table["opt_cpower"]=opt_cpower
        opt_cpower_table["opt_cpower30"]=round(opt_cpower*0.3)
        
        #월별 최대수요, 최적 계약전력 및 최적 최소요금적용전력 비교해서 최적 요금적용전력(opt_bill_aply_pwr) 구하기
        opt_cpower_table["opt_bill_aply_pwr"]=''
        for i in opt_cpower_table.index:
            if opt_cpower_table.at[i, "max_power"]>=opt_cpower_table["opt_cpower30"][0]:
                opt_cpower_table.at[i, "opt_bill_aply_pwr"]=opt_cpower_table.at[i, "max_power"]
            else:
                opt_cpower_table.at[i, "opt_bill_aply_pwr"]=opt_cpower_table.at[i, "opt_cpower30"]
        
        opt_cpower_table["opt_bill_aply_pwr"]=opt_cpower_table["opt_bill_aply_pwr"].astype(float)
        
        
        #월별 최대수요전력, 최소 요금적용전력 반영해 예상 요금적용전력 구하기 
        #이전 달의 최대수요전력, 최소 요금적용전력 고려해서 보수적으로 요금적용전력 산정하기 
        try:
            for i in opt_cpower_table.index:
                if opt_cpower_table.at[i, "opt_bill_aply_pwr"]>=opt_cpower_table.at[i+1, "opt_bill_aply_pwr"]:
                    opt_cpower_table.at[i+1, "opt_bill_aply_pwr"]=opt_cpower_table.at[i, "opt_bill_aply_pwr"]
                else:
                    opt_cpower_table.at[i+1, "opt_bill_aply_pwr"]=opt_cpower_table.at[i+1, "opt_bill_aply_pwr"]
    
        except KeyError:
            pass
    
        
        #요금 절감액 계산하기 = 요금적용전력 절감분 * 기본요금 단가 
        opt_cpower_table["savingCpower"]=opt_cpower_table["now_bill_aply_pwr"]-opt_cpower_table["opt_bill_aply_pwr"]
        opt_cpower_table["save_cost"]=opt_cpower_table["savingCpower"]*BaseCost
        
        # 예상절감액이 양수(+)(계약전력 감소)인 경우 
        if opt_cpower_table["save_cost"].sum()>0:
            change=1
        
        # 예상절감액이 0인 경우 - 계약전력 유지 
        elif opt_cpower_table["save_cost"].sum()==0:
            change=0
        
        # 예상절감액이 음수(-)이고(계약전력 증설), 연 평균 수용률이 70% 미만인 경우
        # 계약전력 유지, 피크관리를 통한 요금적용전력 절감 유도
        elif round(opt_cpower_table["max_power"].mean()/custInfo["cntrPwr"][0]*100,1)<70:
            change=0
            opt_cpower=cpower
            opt_cpower_table["opt_cpower"]=opt_cpower
            opt_cpower_table["opt_cpower30"]=opt_cpower*0.3
            
            #opt_bill_aply_pwr : 현재 계약전력 기준으로 12개월 이전 고려하지 않은 요금적용전력
            for i in opt_cpower_table.index:
                if opt_cpower_table.at[i, "max_power"]>=opt_cpower_table["opt_cpower30"][0]:
                    opt_cpower_table.at[i, "opt_bill_aply_pwr"]=opt_cpower_table.at[i, "max_power"]
                else:
                    opt_cpower_table.at[i, "opt_bill_aply_pwr"]=opt_cpower_table.at[i, "opt_cpower30"]
            
            opt_cpower_table["opt_bill_aply_pwr"]=opt_cpower_table["opt_bill_aply_pwr"].astype(float)
            opt_cpower_table["savingCpower"]=opt_cpower_table["now_bill_aply_pwr"]-opt_cpower_table["opt_bill_aply_pwr"]
            opt_cpower_table["save_cost"]=opt_cpower_table["savingCpower"]*BaseCost
       
        # 예상절감액이 음수(-)이고(계약전력 증설), 연 평균 수용률이 70% 이상인 경우 
        # 계약전력 증설, 요금 증가 예상 
        else:
            change=1
        
        comment=pd.DataFrame({"cntrPwr" : [custInfo["cntrPwr"][0]], #현재 계약전력
                              "billAplyPwrNow" : [opt_cpower_table["billAplyPwr"][len(opt_cpower_table)-1]],    #가장 최근달 요금적용전력
                              "avgDemandkW" : [opt_cpower_table["max_power"].mean()],    #월평균 최대수요전력
                              "avgDemandkWPC" : [round(opt_cpower_table["max_power"].mean()/custInfo["cntrPwr"][0]*100,1)],  #월평균 최대수요전력 수용률
                              "maxDemandkW" : [max(opt_cpower_table["max_power"])], #연간 최대수요전력
                              "maxDemandkWPC" : [round(max(opt_cpower_table["max_power"])/custInfo["cntrPwr"][0]*100,1)],   #연간 최대수요전력 수용률
                              "change" : [change],  #계약전력으로 인한 요금 절감 가능성
                              "optCpower" : [opt_cpower_table["opt_cpower"][0]],    #최적 계약전력
                              "saveCost" : [opt_cpower_table["save_cost"].sum().astype('int64')]})  #최적 계약전력 변경 시 예상 절감액 
        
        
        opt_cpower_table=opt_cpower_table.sort_values('date')
        
        #계약전력 변경 전, 후 요금적용전력 그래프 그리기 
        fig1=plt.figure(figsize=(16,3))
        plt.plot(opt_cpower_table["date"].astype(str), opt_cpower_table["now_bill_aply_pwr"], marker='o', label="변경 전 요금적용전력", color='#B32712')
        plt.plot(opt_cpower_table["date"].astype(str), opt_cpower_table["opt_bill_aply_pwr"], marker='o', label="변경 후 요금적용전력", color='#124EB3')
        plt.plot(opt_cpower_table["date"].astype(str), opt_cpower_table["max_power"], marker='o', label='최대수요', color='black')
        plt.fill_between(opt_cpower_table["date"].astype(str), opt_cpower_table["opt_bill_aply_pwr"], opt_cpower_table["now_bill_aply_pwr"], alpha=0.5, color='#B32712', label="요금절감")
        plt.ylim(opt_cpower_table[["max_power", "billAplyPwr"]].min().min()-50, opt_cpower_table[["max_power", "billAplyPwr"]].max().max()+50)
        plt.ylabel("요금적용전력(kW)", fontsize=14)
        plt.legend(bbox_to_anchor=(0.85,-0.1), ncol=4, fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        fig1.suptitle("예상 절감 비용 : {}만원".format(round(opt_cpower_table["save_cost"].sum()/10000)), fontsize=16)
        # plt.tight_layout()
        plt.grid(False)
        plt.savefig("images\\anlyctrpeak_1.png", bbox_inches='tight',pad_inches=0.4)
        
        
        
        
        ####피크최적화####
        df=hourly
        df['month'] = df['month'].astype("int")
        # 월별계절정보
        info_month=pd.DataFrame({'month':[1,2,3,4,5,6,7,8,9,10,11,12],
                                 'season':['winter', 'winter', 'sf', 'sf', 'sf', 'summer', 'summer', 'summer', 'sf', 'sf', 'winter', 'winter']})
        
        # 계절별 부하시간대 정보(23년 11월 9일 시행 전기요금표 기준)
        info_load=pd.DataFrame({'summer':[1,1,1,1,1,1,1,1,2,2,2,3,2,3,3,3,3,3,2,2,2,2,1,1],
                                'sf':[1,1,1,1,1,1,1,1,2,2,2,3,2,3,3,3,3,3,2,2,2,2,1,1],
                                'winter':[1,1,1,1,1,1,1,1,2,3,3,3,2,2,2,2,3,3,3,2,2,2,1,1]})    # 1: 경부하, 2: 중간부하, 3: 최대부하
        
        info_load = info_load.applymap(str)
        
        # 토요일 부하시간대 정보 수정
        info_load_saturday = info_load.replace('3', '2')

        # 월별 계절 정보 병합
        df = df.merge(info_month, on='month', how='left')
        df.index = hourly.index
        
        # 부하시간대 이름 매핑 함수
        def map_load_name(row, is_saturday=False):
            if is_saturday:
                return info_load_saturday.loc[row['hour'], row['season']]
            else:
                return info_load.loc[row['hour'], row['season']]
        
        # 부하시간대 이름 추가
        df['day'] = df.index.weekday
        df['hour'] = df.index.hour
        df['load_nm'] = df.apply(lambda row: map_load_name(row, row['day'] == 5), axis=1)
        
        # 평일 중간부하-최대부하 시간대만 필터링 
        df_weekday=df.drop(df[(df['day']==5)|(df['day']==6)].index)
        df_saturday=df[df['day']==5]
        df_sunday=df[df['day']==6]
        df_sunday['load_nm']='1'
        
        # 평일 중간부하-최대부하 시간대 수요전력만 고려: 요금적용전력에 영향
        df=df_weekday.drop(df_weekday[df_weekday['load_nm']=='1'].index)

        h=plt.hist(df["demandkW"], bins=70)
        hdf=pd.DataFrame(columns=["max_elect","counts","diff"])
        hdf["max_elect"]=h[1]
        first=np.array([0])
        hdf["counts"]=np.append(h[0],first)
        hdf["diff"]=np.append(first, np.append(-np.diff(h[0]),0))
    
        Final_Goal=fun.Find_Goal(hdf, df, GoalPeak, GoalTime) #목표 피크전력
        Final_Time=fun.Find_Time(hdf, df, GoalPeak, GoalTime) #관리해야 할 시간
        
        #ManagePower : 전년대비 관리해야 할 피크전력량 
        ManagePower=round(max(df["demandkW"])-Final_Goal)   
        
        
        #Final_Goal 이상의 전력량 표시해주기 
        df["dkW_opt"]=''
        for b in df.index:
            if df.demandkW[b]>Final_Goal:
                df.at[b, 'dkW_opt']=float(Final_Goal)
            else:
                df.at[b, 'dkW_opt']=df.at[b, 'demandkW']
        
        df.dkW_opt=df.dkW_opt.astype(float)
        
        
        #현재 최대수요전력이 계약전력의 30% 미만인 경우, 계약전력 변경 후 피크최적화 필요 
        if df["demandkW"].max()<cpower30:
            cpower_change=1 #계약전력 변경 선행 필요
            opt_cpower_peak_table=opt_cpower_table.copy()   #opt_cpower_peak_table : 계약전력 변경, 피크최적화 모두 반영 
            
            #opt_max_power : 피크최적화 이후, 월별 최대수요전력 < 최적피크
            for i in opt_cpower_peak_table.index:    
                if opt_cpower_peak_table.at[i, "max_power"]>Final_Goal:
                    opt_cpower_peak_table.at[i, "opt_max_power"]=Final_Goal
                else:
                    opt_cpower_peak_table.at[i, "opt_max_power"]=opt_cpower_peak_table.at[i, "max_power"]
    
            #요금적용전력 재산정
            opt_cpower_peak_table["opt_bill_aply_pwr"]=''
            for i in range(0, len(opt_cpower_peak_table)):
                if opt_cpower_peak_table["opt_max_power"].iloc[i]>=opt_cpower_peak_table["opt_cpower30"][0]:
                    opt_cpower_peak_table["opt_bill_aply_pwr"].iloc[i]=opt_cpower_peak_table["opt_max_power"].iloc[i]
                else:
                    opt_cpower_peak_table["opt_bill_aply_pwr"].iloc[i]=opt_cpower_peak_table["opt_cpower30"].iloc[i]
    
            opt_cpower_peak_table["opt_bill_aply_pwr"]=opt_cpower_peak_table["opt_bill_aply_pwr"].astype(float)
    
            #달마다 밀어가면서 요금적용전력 산정하기
            try:
                for i in opt_cpower_peak_table.index:
                    if opt_cpower_peak_table.at[i, "opt_bill_aply_pwr"]>=opt_cpower_peak_table.at[i+1, "opt_bill_aply_pwr"]:
                        opt_cpower_peak_table.at[i+1, "opt_bill_aply_pwr"]=opt_cpower_peak_table.at[i, "opt_bill_aply_pwr"]
                    else:
                        opt_cpower_peak_table.at[i+1, "opt_bill_aply_pwr"]=opt_cpower_peak_table.at[i+1, "opt_bill_aply_pwr"]
    
            except KeyError:
                pass
    
            #예상 요금절감 계산하기 
            opt_cpower_peak_table["savingCpower"]=opt_cpower_peak_table["now_bill_aply_pwr"]-opt_cpower_peak_table["opt_bill_aply_pwr"]
            opt_cpower_peak_table["save_cost"]=opt_cpower_peak_table["savingCpower"]*BaseCost   
            
            opt_cpower_peak_table=opt_cpower_peak_table.sort_values("date")
            
            #plot
            fig2=plt.figure(figsize=(12,12))
            ax1=fig2.add_subplot(3,1,1)
            n, bins, patches=plt.hist(df["demandkW"], bins=70, rwidth=0.9)
            
            #피크관리가 필요한 부분 표시하기 
            for j in range(bisect.bisect_left(hdf["max_elect"], Final_Goal), len(hdf)-1): 
                patches[j].set_fc('red')
            plt.bar_label(patches, fontsize=8, color='#5C8FE6')
            plt.vlines(Final_Goal, 0, hdf["counts"].max(), color='red', linestyles='--')
            plt.title("피크최적화 목표수립\n현재 : {}kW, 추천 : {}kW, 관리시간 : {}시간".format(max(df["demandkW"]), max(df["dkW_opt"]), int(Final_Time)), fontsize=16)
            plt.xlabel("전력수요(kW)", fontsize=14)
            plt.ylabel("빈도수", fontsize=14)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            current_x=plt.gca().get_xticks()
            plt.gca().set_xticklabels(['{:,.0f}'.format(x) for x in current_x])
            current_values = plt.gca().get_yticks()
            plt.gca().set_yticklabels(['{:,.0f}'.format(x) for x in current_values])
            plt.grid(True)
            
            ax2=fig2.add_subplot(3,1,2)
            plt.scatter(df["hour"].astype(int), df["demandkW"], c='#B32712', s=40, edgecolors='black', label="관리 전")
            plt.scatter(df["hour"].astype(int), df["dkW_opt"], s=40, edgecolors='black', c='#5C8FE6', label="관리 후")
            plt.hlines(Final_Goal, 0, 23, color='red', linestyle='--')
            plt.hlines(df["demandkW"].max(), 0, 23, color='red', linestyle='--')
            plt.title("피크최적화 관리", fontsize=16)
            plt.ylabel("전력수요(kW)", fontsize=14)
            plt.xticks([0,4,8,12,16,20,24], labels=['00시','04시', '08시', '12시', '16시', '20시', '24시'], fontsize=14)
            plt.legend(bbox_to_anchor=(0.64,-0.1), ncol=2, fontsize=14)
            plt.ylim(df["demandkW"].min(), df["demandkW"].max()*1.05)
            plt.yticks(fontsize=14)
            current_values = plt.gca().get_yticks()
            plt.gca().set_yticklabels(['{:,.0f}'.format(x) for x in current_values])
            plt.grid(True)
            
            ax3=fig2.add_subplot(3,1,3)
            plt.plot(opt_cpower_peak_table.date, opt_cpower_peak_table["now_bill_aply_pwr"], marker='^', label="계약전력 변경 전 요금적용전력", color='#B32712')
            plt.plot(opt_cpower_peak_table.date, opt_cpower_peak_table["opt_bill_aply_pwr"], marker='^', label="계약전력 변경 후 요금적용전력", color='#124EB3')
            plt.plot(opt_cpower_peak_table.date, opt_cpower_peak_table["max_power"], marker='o', label="피크최적화 이전 최대수요전력", color='black')
            plt.plot(opt_cpower_peak_table.date, opt_cpower_peak_table["opt_max_power"], marker='o', label="피크최적화 이후 최대수요전력", color='#B32712')
            plt.ylim(opt_cpower_table[["max_power", "billAplyPwr"]].min().min()-50, opt_cpower_table[["max_power", "billAplyPwr"]].max().max()+50)
            plt.fill_between(opt_cpower_peak_table.date, opt_cpower_peak_table["opt_bill_aply_pwr"], opt_cpower_peak_table["now_bill_aply_pwr"], alpha=0.5, color='#B32712', label="요금절감")
            plt.ylabel("요금적용전력(kW)", fontsize=14)
            plt.legend(bbox_to_anchor=(0.85,-0.1), ncol=3, fontsize=14)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.title("계약전력 변경 + 피크최적화 시 예상 절감 비용 : {}만원".format(round(opt_cpower_peak_table["save_cost"].sum()/10000)), fontsize=16)
            
            fig2.tight_layout()
            plt.savefig("images\\anlyctrpeak_2.png", bbox_inches='tight',pad_inches=0.4)
            
            
            
        else:
            cpower_change=0 #계약전력 변경 선행 불필요
            #현재 최대수요전력이 계약전력의 30% 이상인 경우, 계약전력 변경과 피크최적화 상관없음 
            opt_cpower_peak_table=opt_cpower_table.copy()
            
            #opt_max_power : 피크최적화 이후, 월별 최대수요전력 < 최적피크
            for i in opt_cpower_peak_table.index:    
                if opt_cpower_peak_table.at[i, "max_power"]>Final_Goal:
                    opt_cpower_peak_table.at[i, "opt_max_power"]=Final_Goal
                else:
                    opt_cpower_peak_table.at[i, "opt_max_power"]=opt_cpower_peak_table.at[i, "max_power"]
    
            #요금적용전력 재산정
            opt_cpower_peak_table["opt_bill_aply_pwr"]=''
            for i in range(0, len(opt_cpower_peak_table)):
                if opt_cpower_peak_table["opt_max_power"].iloc[i]>=opt_cpower_peak_table["opt_cpower30"][0]:
                    opt_cpower_peak_table["opt_bill_aply_pwr"].iloc[i]=opt_cpower_peak_table["opt_max_power"].iloc[i]
                else:
                    opt_cpower_peak_table["opt_bill_aply_pwr"].iloc[i]=opt_cpower_peak_table["opt_cpower30"].iloc[i]
    
            opt_cpower_peak_table["opt_bill_aply_pwr"]=opt_cpower_peak_table["opt_bill_aply_pwr"].astype(float)
    
            #달마다 밀어가면서 요금적용전력 산정하기
            try:
                for i in opt_cpower_peak_table.index:
                    if opt_cpower_peak_table.at[i, "opt_bill_aply_pwr"]>=opt_cpower_peak_table.at[i+1, "opt_bill_aply_pwr"]:
                        opt_cpower_peak_table.at[i+1, "opt_bill_aply_pwr"]=opt_cpower_peak_table.at[i, "opt_bill_aply_pwr"]
                    else:
                        opt_cpower_peak_table.at[i+1, "opt_bill_aply_pwr"]=opt_cpower_peak_table.at[i+1, "opt_bill_aply_pwr"]
    
            except KeyError:
                pass
    
            #예상 요금절감 계산하기 
            opt_cpower_peak_table["savingCpower"]=opt_cpower_peak_table["now_bill_aply_pwr"]-opt_cpower_peak_table["opt_bill_aply_pwr"]
            opt_cpower_peak_table["save_cost"]=opt_cpower_peak_table["savingCpower"]*BaseCost   
            
            opt_cpower_peak_table=opt_cpower_peak_table.sort_values("date")
    
            #plot
            fig2=plt.figure(figsize=(12,12))
            ax1=fig2.add_subplot(3,1,1)
            n, bins, patches=plt.hist(df["demandkW"], bins=70, rwidth=0.9)
            
            #피크관리가 필요한 부분 표시하기 
            for j in range(bisect.bisect_left(hdf["max_elect"], Final_Goal), len(hdf)-1): 
                patches[j].set_fc('red')
            plt.bar_label(patches, fontsize=8, color='#5C8FE6')
            plt.vlines(Final_Goal, 0, hdf["counts"].max(), color='red', linestyles='--')
            plt.title("피크최적화 목표수립\n현재 : {}kW, 추천 : {}kW, 관리시간 : {}시간".format(max(df["demandkW"]), max(df["dkW_opt"]), int(Final_Time)), fontsize=16)
            plt.xlabel("전력수요(kW)", fontsize=14)
            plt.ylabel("빈도수", fontsize=14)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            current_x=plt.gca().get_xticks()
            plt.gca().set_xticklabels(['{:,.0f}'.format(x) for x in current_x])
            current_values = plt.gca().get_yticks()
            plt.gca().set_yticklabels(['{:,.0f}'.format(x) for x in current_values])
            plt.grid(True)
            
            ax2=fig2.add_subplot(3,1,2)
            plt.scatter(df["hour"].astype(int), df["demandkW"], c='#B32712', s=40, edgecolors='black', label="관리 전")
            plt.scatter(df["hour"].astype(int), df["dkW_opt"], s=40, edgecolors='black', c='#5C8FE6', label="관리 후")
            plt.hlines(Final_Goal, 0, 23, color='red', linestyle='--')
            plt.hlines(df["demandkW"].max(), 0, 23, color='red', linestyle='--')
            plt.title("피크최적화 관리", fontsize=16)
            plt.ylabel("전력수요(kW)", fontsize=14)
            plt.xticks([0,4,8,12,16,20,24], labels=['00시','04시', '08시', '12시', '16시', '20시', '24시'], fontsize=14)
            plt.legend(bbox_to_anchor=(0.64,-0.1), ncol=2, fontsize=14)
            plt.ylim(df["demandkW"].min(), df["demandkW"].max()*1.05)
            plt.yticks(fontsize=14)
            current_values = plt.gca().get_yticks()
            plt.gca().set_yticklabels(['{:,.0f}'.format(x) for x in current_values])
            plt.grid(True)
            
            ax3=fig2.add_subplot(3,1,3)
            plt.plot(opt_cpower_peak_table.date, opt_cpower_peak_table["now_bill_aply_pwr"], marker='^', label="계약전력 변경 전 요금적용전력", color='#B32712')
            plt.plot(opt_cpower_peak_table.date, opt_cpower_peak_table["opt_bill_aply_pwr"], marker='^', label="계약전력 변경 후 요금적용전력", color='#124EB3')
            plt.plot(opt_cpower_peak_table.date, opt_cpower_peak_table["max_power"], marker='o', label="피크최적화 이전 최대수요전력", color='black')
            plt.plot(opt_cpower_peak_table.date, opt_cpower_peak_table["opt_max_power"], marker='o', label="피크최적화 이후 최대수요전력", color='#B32712')
            plt.ylim(opt_cpower_table[["max_power", "billAplyPwr"]].min().min()-50, opt_cpower_table[["max_power", "billAplyPwr"]].max().max()+50)
            plt.fill_between(opt_cpower_peak_table.date, opt_cpower_peak_table["opt_bill_aply_pwr"], opt_cpower_peak_table["now_bill_aply_pwr"], alpha=0.5, color='#B32712', label="요금절감")
            plt.ylabel("요금적용전력(kW)", fontsize=14)
            plt.legend(bbox_to_anchor=(0.85,-0.1), ncol=3, fontsize=14)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.title("계약전력 변경 + 피크최적화 시 예상 절감 비용 : {}만원".format(round(opt_cpower_peak_table["save_cost"].sum()/10000)), fontsize=16)
            
            fig2.tight_layout()
            plt.savefig("images\\anlyctrpeak_2.png", bbox_inches='tight',pad_inches=0.4)
            
    
            
        #anlyPeakTable
        anlyPeakTable=pd.DataFrame({'cpower_change':[cpower_change],
                                    'curPeak':[round(max(df["demandkW"]))],
                                    'optPeak':[Final_Goal],
                                    'manPerc' : [round(Final_Goal/max(df["demandkW"])*100,2)],
                                    'mngkW':[ManagePower],
                                    'mngTime':[Final_Time],
                                    'expBill':[opt_cpower_peak_table["save_cost"].sum()]}).astype('int64')

    else:
        print("계약전력 20kW 미만으로, 요금적용전력이 계약전력으로 고정되는 건물입니다.")
        anlyCntrTable=pd.DataFrame()
        comment='0'
        fig1='0'
        fig2='0'
        anlyPeakTable=pd.DataFrame()
    
    
    return anlyCntrTable, comment, fig1, fig2, anlyPeakTable


#### 5_역률분석
def anlyPF(custNo, pw, start, end):
    #DB conn & query
    db = DBConnection()
    db.connect()
    
    #전력단가 불러오기 
    sql="""SELECT * FROM ereport.sei_cost"""
    info_costn=db.execute_query(sql)
    
    #월별 청구요금 테이블에서 역률 정보 가져오기 
    sql="""SELECT * FROM ereport.sei_bill WHERE "custNo"='{}' AND "billYm" BETWEEN '{}' AND '{}'""".format(custNo, start[0:7], end[0:7])
    real_time_info=db.execute_query(sql)
    real_time_info=real_time_info.rename(columns={'billYm':'date'})
    
    ####파워플래너 크롤링####
    getData = pp.powerPlanner(custNo, pw, '')
    
    #분석 시작년도와 종료년도가 동일한 경우
    if start[0:4]==end[0:4]:
        temp=getData.getPowerFactor(start[0:4])
        #크롤링 데이터 형식 맞추기         
        temp_new=pd.DataFrame(columns={'date', 'jnPwrfact', 'jiPwrfact'}, index=[0,1,2,3,4,5,6,7,8,9,10,11])
        for i in temp_new.index:
            temp_new.at[i, 'date']=start[0:4]+'.{}'.format(i+1)
        
        for i in temp.index:
            #1월~6월까지의 데이터 
            temp_new.at[i, 'jiPwrfact']=temp.iloc[i,[1]][0]
            temp_new.at[i, 'jnPwrfact']=temp.iloc[i,[2]][0]
            #7월~12월까지의 데이터
            temp_new.at[i+6, 'jiPwrfact']=temp.iloc[i,[4]][0]
            temp_new.at[i+6, 'jnPwrfact']=temp.iloc[i,[5]][0]
        
        #크롤링 데이터 기간 real_time_info와 맞추기 
        temp_new=temp_new[int(real_time_info['date'][len(real_time_info)-1][5:7])-1:int(real_time_info['date'][0][5:7])]
        temp_new=temp_new.sort_index(ascending=False).reset_index(drop=True)
        
        real_time_info=pd.concat([real_time_info.drop(['jiPwrfact', 'jnPwrfact'], axis=1), temp_new.drop(['date'], axis=1).astype(float)], axis=1)
        
    #분석 시작년도와 종료년도가 동일하지 않은 경우
    else:
        temp1=getData.getPowerFactor(start[0:4])
        temp2=getData.getPowerFactor(end[0:4])
        #크롤링 데이터 형식 맞추기         
        temp_new1=pd.DataFrame(columns={'date', 'jnPwrfact', 'jiPwrfact'}, index=[0,1,2,3,4,5,6,7,8,9,10,11])
        temp_new2=pd.DataFrame(columns={'date', 'jnPwrfact', 'jiPwrfact'}, index=[0,1,2,3,4,5,6,7,8,9,10,11])
        
        for i in temp_new1.index:
            temp_new1.at[i, 'date']=start[0:4]+'.{}'.format(i+1)
            temp_new2.at[i, 'date']=end[0:4]+'.{}'.format(i+1)
        
        for i in temp1.index:
            #1월~6월까지의 데이터 
            temp_new1.at[i, 'jiPwrfact']=temp1.iloc[i,[1]][0]
            temp_new1.at[i, 'jnPwrfact']=temp1.iloc[i,[2]][0]
            temp_new2.at[i, 'jiPwrfact']=temp2.iloc[i,[1]][0]
            temp_new2.at[i, 'jnPwrfact']=temp2.iloc[i,[2]][0]
            #7월~12월까지의 데이터
            temp_new1.at[i+6, 'jiPwrfact']=temp1.iloc[i,[4]][0]
            temp_new1.at[i+6, 'jnPwrfact']=temp1.iloc[i,[5]][0]
            temp_new2.at[i+6, 'jiPwrfact']=temp2.iloc[i,[4]][0]
            temp_new2.at[i+6, 'jnPwrfact']=temp2.iloc[i,[5]][0]
            
        temp_new=pd.concat([temp_new1, temp_new2], axis=0, ignore_index=True)
        #크롤링 데이터 기간 real_time_info와 맞추기 
        temp_new=temp_new[int(real_time_info['date'][len(real_time_info)-1][5:7])-1:int(real_time_info['date'][0][5:7])+12]
        temp_new=temp_new.sort_index(ascending=False).reset_index(drop=True)
        
        real_time_info=pd.concat([real_time_info.drop(['jiPwrfact', 'jnPwrfact'], axis=1), temp_new.drop(['date'], axis=1).astype(float)], axis=1)


    ##### 분석(역률관리)
    #지상역률(lagg_pwrfact) 관리 기준 90%, 평균역률 90% 미만인 경우 미달하는 역률 60%까지 1%당 0.2%추가, 90% 초과하는 경우 역률 95%까지 1%당 0.2% 감액
    real_time_info["jiPwrfact_re"]=''
    for i in range(0, len(real_time_info)):
        if real_time_info["jiPwrfact"].iloc[i]>95:
            real_time_info.at[i, "jiPwrfact_re"]=95
        elif real_time_info["jiPwrfact"].iloc[i]==0: #간혹 지상역률 90% 이상이 0으로 찍히는 경우 있음 
            real_time_info.at[i, "jiPwrfact"]=95    #95로 우선 처리 
            real_time_info.at[i, "jiPwrfact_re"]=95
        elif real_time_info["jiPwrfact"].iloc[i]<60:
            real_time_info.at[i, "jiPwrfact_re"]=60
        else:
            real_time_info.at[i, "jiPwrfact_re"]=real_time_info.at[i, "jiPwrfact"]
        
    real_time_info["jiPwrfact_re_cal"]=90-real_time_info["jiPwrfact_re"]
    real_time_info["jiPwrfact_cost"]=real_time_info["jiPwrfact_re_cal"]*0.002*real_time_info["baseBill"] #기본요금 추가금
    real_time_info["jiPwrfact_cost"]=real_time_info["jiPwrfact_cost"].astype(float).round()


    #진상역률(lead_pwrfact) 관리 기준 95%, 평균역률 95% 미만인 경우 미달하는 역률 1% 당 0.2% 추가
    real_time_info["jnPwrfact_re"]=''
    for i in range(0,len(real_time_info)):
        if real_time_info["jnPwrfact"].iloc[i]>95: 
            real_time_info.at[i, "jnPwrfact_re"]=95
        elif real_time_info["jnPwrfact"].iloc[i]==0: #간혹 진상역률 95% 이상이 0으로 찍히는 경우 있음 
            real_time_info.at[i, "jnPwrfact"]=95
            real_time_info.at[i, "jnPwrfact_re"]=95
        elif real_time_info["jnPwrfact"].iloc[i]<60:
            real_time_info.at[i, "jnPwrfact_re"]=60
        else:
            real_time_info.at[i, "jnPwrfact_re"]=real_time_info.at[i, "jnPwrfact"]

    real_time_info["jnPwrfact_re_cal"]=95-real_time_info["jnPwrfact"]
    real_time_info["jnPwrfact_cost"]=real_time_info["jnPwrfact_re_cal"].astype(float)*0.002*real_time_info["baseBill"] #기본요금 추가금 
    real_time_info["jnPwrfact_cost"]=real_time_info["jnPwrfact_cost"].astype(float).round()

    real_time_info["save_cost"]=real_time_info["jiPwrfact_cost"]+real_time_info["jnPwrfact_cost"] #역률관리가 됬더라면 절약했을 요금
    
    jiStan=90 #지상역률 기준
    jnStan=95 #진상역률 기준
    real_time_info["laggPFVar"]=real_time_info["jiPwrfact"]-jiStan
    real_time_info["leadPFVar"]=real_time_info["jnPwrfact"]-jnStan
    
    anlyPF=real_time_info[["date", "jiPwrfact", "laggPFVar", "jnPwrfact", "leadPFVar", 'jiPwrfact_cost', 'jnPwrfact_cost', "save_cost"]]
    anlyPF=anlyPF.rename(columns={'jiPwrfact' : 'jiPF', 'jnPwrfact' : 'jnPF', 'save_cost' : 'pfBill'})
    anlyPF["jiStan"]=jiStan
    anlyPF["jnStan"]=jnStan
    
    anlyPF=anlyPF.sort_values("date").reset_index(drop=True)
    
    #comment
    comment=pd.DataFrame(columns={'jiChange', 'jnChange', 'savedCost', 'savingCost'})
    #분석기간 중 지상역률이 90미만인 달이 한번이라도 있는 경우, 지상역률 개선 필요
    if len(real_time_info[real_time_info["jiPwrfact_re_cal"]>0])>0:   
        comment.at[0, "jiChange"]=1
    else:
        comment.at[0, "jiChange"]=0
    
    #분석기간 중 진상역률이 95미만인 달이 한번이라도 있는 경우, 진상역률 개선 필요
    if len(real_time_info[real_time_info["jnPwrfact_re_cal"]>0])>0:
        comment.at[0, "jnChange"]=1
    else:
        comment.at[0, "jnChange"]=0
    
    #앞으로 관리해서 받을 수 있는 할인 금액(savingCost)
    if len(real_time_info[real_time_info["save_cost"]>0])>0:
        comment.at[0, 'savingCost']=real_time_info[real_time_info["save_cost"]>0]["save_cost"].sum()
    else:
        comment.at[0, 'savingCost']=0
        
    #여태 할인받은 금액(savedCost)
    if len(real_time_info[real_time_info["save_cost"]<0])>0:
        comment.at[0, 'savedCost']=real_time_info[real_time_info['save_cost']<0]['save_cost'].sum()
    else:
        comment.at[0, 'savingCost']=0
        
    return anlyPF, comment


#### 6_설비효율화
def anlyEfficient(custNo, start, end, location):
    plt.rcParams["font.family"]='Malgun Gothic'
    plt.rcParams['axes.unicode_minus']=False    
    
    ####db connect
    db = DBConnection()
    db.connect()
    
    #15분 단위 전력 데이터 불러오기 - minute : 결측치 처리 안 되어 있음, 다계기 없음(과거데이터)
    sql="""SELECT * FROM ereport.sei_usekwh WHERE "custNo"= '{}' AND "mrYmd" BETWEEN '{}' AND '{}'""".format(custNo, start, end)
    minute=db.execute_query(sql, dtype={'mrYmd':'datetime64[ns]'})
    minute.set_index("mrYmd", inplace=True)
    
    #결측치 0처리 
    r=pd.date_range(start=start, end=pd.to_datetime(end)+dt.timedelta(0,-900,0), freq='15min', name='mrYmd')
    data_minute=pd.DataFrame(index=r)
    data_minute=data_minute.join(minute)
    data_minute.fillna(0, inplace=True)
 
    #15분 -> 1일 가공하기 
    dailyEnergy=pd.DataFrame()
    dailyEnergy['usekWh']=data_minute.resample('1D').sum()['pwrQty']

    #기상 데이터 불러오기 
    # sql="""SELECT * FROM ereport.sei_weather WHERE "kmaNm" = '{}' AND tm BETWEEN '{}' AND '{}'""".format(location, start, pd.to_datetime(end)+dt.timedelta(0,-900,0))
    # weather=db.execute_query(sql, dtype={'tm':'datetime64[ns]'})
    weather=pd.read_excel("weather.xlsx", dtype={'tm':'datetime64[ns]'})
    weather=weather.rename(columns={'tm':'datetime', 'ta':'Temperature'})
    
    #기상데이터 전처리
    weather=weather[["datetime", "Temperature"]]
    weather=weather.set_index(pd.to_datetime(weather['datetime']), drop=True)
    weather=weather.loc[~weather.index.duplicated(keep='first')]
    weather=weather.reindex(pd.date_range(start=start, end=pd.to_datetime(end)+dt.timedelta(0,-900,0), freq = '1H')).fillna(method='ffill')
    dailyWeather=pd.DataFrame(round(weather['Temperature'].resample('24H').mean(),1))
    
    # dailyData : 냉난방효율화 분석 input data : 일일 전력사용량, 일 평균 기온
    dailyData = pd.concat([dailyEnergy, dailyWeather], axis=1)
    dailyData["weekday"]=dailyData.index.weekday
    dailyData['weekday']=dailyData['weekday'].replace(to_replace=[0,1,2,3,4,5,6], value=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])

    #Anormaly : 일별 전력사용량 기준으로 이상가동일 추정 
    result=seasonal_decompose(dailyData["usekWh"], model='additive')
    constance=0.5
    
    Q1 = np.percentile(result.resid.dropna(), 25)
    Q3 = np.percentile(result.resid.dropna(), 75)
    IQR = Q3 - Q1 
    IQR = IQR if IQR > 0 else -1*IQR
    lower = Q1 - constance*IQR
    higher = Q3 + constance*IQR
    
    #IQR 기준으로 이상치 판단하기 
    for i in result.resid.dropna().index:
        if result.resid.dropna().loc[i] < lower or  result.resid.dropna().loc[i] > higher:
            dailyData.at[i,"Anormaly"]=1
        else:
            dailyData.at[i,"Anormaly"]=0
            
    #결측값에 대해 이상치 판단하기 
    for i in dailyData[dailyData["Anormaly"].isna()].index:
        dailyData.at[i, "Anormaly"]=1
    
    holidayList=pytimekr.holidays(year=int(start[0:4]))+pytimekr.holidays(year=int(end[0:4]))
    holidayList=list(set(holidayList))

    dailyData['holiday']=''
    for i in dailyData.index:
        #공휴일인지 여부 판단하기 
        if dailyData.loc[i].name in holidayList:
            dailyData.at[i, 'holiday']='True'
        else:
            dailyData.at[i, 'holiday']='False'
        
        #Anormaly 중 공휴일 제외하기 
        if dailyData.loc[i]['holiday']=='True' and dailyData.loc[i]['Anormaly']==1:
            dailyData.at[i, "Anormaly"]==0
            dailyData.at[i, "usekWhAft"]=dailyData.at[i, "usekWh"]
        elif dailyData.loc[i]['holiday']=='Fasle' and dailyData.loc[i]['Anormaly']==1:
            dailyData.at[i, "usekWhAft"]=dailyData.usekWh.mean()
        else:
            dailyData.at[i, "usekWhAft"]=dailyData.at[i, "usekWh"]

    ####주말&공휴일 발라내기
    weekendList=['Sat', 'Sun']
    dailyData['operationDay']=''
    for i in dailyData.index:
        if dailyData.loc[i].name in holidayList or dailyData.loc[i].weekday in weekendList:
            dailyData.at[i, 'operationDay']=0
        else:
            dailyData.at[i, 'operationDay']=1
    
    #평일과 주말&공휴일 사용량의 유사성 비교해보기 
    #평일과 주말&공휴일 사용량 유사성이 높은 경우에는 CPR 각각 하고,  낮은 경우에는 평일에만 하기 
    ks_statistic, p_value=stats.ks_2samp(dailyData[dailyData['operationDay']==1]["usekWh"], dailyData[dailyData['operationDay']==0]["usekWh"])
    
    ####유사성에 따라 PCR 그래프 그리기 
    similarity=''
    if p_value < 0.05:
        operation_days = [0,1]  # 운영일과 비운영일
        similarity='0'
        print("건물 운영일과 비운영일의 분포가 유의미하게 달라, 각각의 CPM을 생성합니다.")
    else:
        dailyData['operationDay']=1
        operation_days = [1]  # 운영일
        similarity='1'
        print("건물 운영일과 비운영일의 분포가 유사하므로, 분석기간 전체를 건물 운영일로 가정합니다.")
    
    fig1 = plt.figure(figsize=(16,10))
    fig1.set_facecolor('white')
    ax = fig1.add_subplot()
    
    for op_day in operation_days:
        dailyDataFiltered1 = dailyData[dailyData['operationDay'] == op_day].copy()
        dailyDataFiltered = dailyDataFiltered1[dailyDataFiltered1["Anormaly"]==0]
        dailyAnormaly = dailyDataFiltered1[dailyDataFiltered1["Anormaly"]==1]
        
        x = deepcopy(dailyDataFiltered['Temperature'])
        y = deepcopy(dailyDataFiltered['usekWh'])
        X = np.expand_dims(x, axis=1)
        order = 1   # 다항차수
    
        final_knots = [18, 24]  # 난방 시작온도 : 18도 / 냉방 시작온도 : 24도 
    
        try:    
            pr = PiecewiseRegression(final_knots, order=order).fit(X, y)
        except ValueError:
            weather=weather.reindex(pd.date_range(start=start, end=end, freq = '1H')).fillna(method='ffill')
            dailyWeather=pd.DataFrame(weather['Temperature'].resample('24H').mean())
            dailyData = pd.concat([dailyEnergy, dailyWeather], axis=1)
            dailyDataFiltered = dailyData[(np.abs(stats.zscore(dailyData.usekWh)) < 3)]  # Z-score outlier remove
            x = deepcopy(dailyDataFiltered['Temperature'])
            y = deepcopy(dailyDataFiltered['usekWh'])
            X = np.expand_dims(x, axis=1)  
            order = 1   #다항 차수
            pr = PiecewiseRegression(final_knots, order=order).fit(X, y)
    
        dailyDataCool = dailyDataFiltered[dailyDataFiltered['Temperature'] >= 24]
        coolR = stats.pearsonr(dailyDataCool.Temperature, dailyDataCool.usekWh)
        dailyDataHeat = dailyDataFiltered[dailyDataFiltered['Temperature'] <= 18]
        heatR = stats.pearsonr(dailyDataHeat.Temperature, dailyDataHeat.usekWh)
        
        ax.scatter(dailyAnormaly["Temperature"], dailyAnormaly["usekWh"], s=200, color='red', label='이상운영일' if op_day == 1 else '')
        ax.scatter(x, y, s=200, label='운영일' if op_day == 1 else '비운영일', color='#007380' if op_day == 1 else 'orange')
        
        for d in range(len(final_knots)+1):
            add_vline = False
            if d == 0:
                idx = x<=final_knots[d]
                add_vline=True
            elif d == len(final_knots):
                idx = x>=final_knots[len(final_knots)-1]
            else:
                idx = (x<=final_knots[d])&(x>=final_knots[d-1])
                add_vline=True
            if add_vline:    
                ax.axvline(final_knots[d], linestyle='--', color='grey')
            ax.plot(x[idx], pr.predict(X)[idx], color='black')
                
        # 각 데이터 결과에 대한 plt.text를 루프 안에서 설정
        if op_day == 0:
            text_y = dailyDataFiltered['usekWh'].min()*1.2
            plt.text(15, text_y, "$R_{heat}$ = %.2f" % heatR[0], color='#B32712', fontsize=20, verticalalignment='bottom')
            plt.text(22, text_y, "$R_{cool}$ = %.2f" % coolR[0], color='#124EB3', fontsize=20, verticalalignment='bottom')
        else:
            text_y = dailyDataFiltered['usekWh'].max()*0.8 
            plt.text(15, text_y, "$R_{heat}$ = %.2f" % heatR[0], color='#B32712', fontsize=20, verticalalignment='bottom')
            plt.text(22, text_y, "$R_{cool}$ = %.2f" % coolR[0], color='#124EB3', fontsize=20, verticalalignment='bottom')

        # # 점과 겹치지 않는 위치에 텍스트 배치
        # ax.text(10, text_y, "$R_{heat}$ = %.2f" % heatR[0], color='#B32712', fontsize=20, verticalalignment='bottom')
        # ax.text(20, text_y, "$R_{cool}$ = %.2f" % coolR[0], color='#124EB3', fontsize=20, verticalalignment='bottom')
        
        #운영일에 한해 pr 계수값 뽑기
        #return 값 정의 -1
        if op_day ==1:
            anlyEfficientGrph=round(dailyDataFiltered,1)
            anlyEfficientRgr=[round(pr.coef_[1],1), round(pr.coef_[0],1), round(pr.coef_[3],1), round(pr.coef_[2],1), round(pr.coef_[5],1), round(pr.coef_[4],1)]
            Rheat=round(heatR[0],2)
            Rcool=round(coolR[0],2)
        else:
            pass

    plt.ylabel('일일 전력사용량(kWh)', fontsize=25)
    plt.xlabel('외기온도(˚C)', fontsize=25)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    current_values = plt.gca().get_yticks()
    plt.gca().set_yticklabels(['{:,.0f}'.format(x) for x in current_values])
    
    plt.legend(fontsize=20)
    
    plt.savefig('images\\anlypf_1.png', bbox_inches='tight',pad_inches=0.4)

    
    #절감액 도출 여부 
    if (Rheat>=-0.2)&(Rcool<0.2):
        save=0  #난방기기, 냉방기기 효율화 불가 
    elif (Rheat<-0.2)&(Rcool<0.2):
        save=1  #난방기기만 효율화
    elif (Rheat>=-0.2)&(Rcool>=0.2):
        save=2  #냉방기기만 효율화
    else:
        save=3  #난방기기, 냉방기기 모두 효율화 
    
    #기저부하 : CPM으로 도출한 Cooling/Heating change point에서의 전력사용량 평균 
    baseload=(pr.coef_[0]+pr.coef_[1]*pr.knots[0]+pr.coef_[4]+pr.coef_[5]*pr.knots[1])/2
    
    #일별 에너지 사용량 기저/난방/냉방으로 분리하기 
    for i in dailyDataFiltered.index:
        if dailyDataFiltered.at[i, "Temperature"]<18:   
            if dailyDataFiltered.at[i, "usekWh"]<baseload:  
                dailyDataFiltered.at[i, "HeatingE"]=0   
                dailyDataFiltered.at[i, "CoolingE"]=0   
                dailyDataFiltered.at[i, "BaseE"]=dailyDataFiltered.at[i, "usekWh"]
            else:
                dailyDataFiltered.at[i, "HeatingE"]=dailyDataFiltered.at[i, "usekWh"]-baseload
                dailyDataFiltered.at[i, "CoolingE"]=0
                dailyDataFiltered.at[i, "BaseE"]=baseload
        elif dailyDataFiltered.at[i, "Temperature"]>24:
            if dailyDataFiltered.at[i, "usekWh"]<baseload:
                dailyDataFiltered.at[i, "CoolingE"]=0
                dailyDataFiltered.at[i, "HeatingE"]=0
                dailyDataFiltered.at[i, "BaseE"]=dailyDataFiltered.at[i, "usekWh"]
            else:
                dailyDataFiltered.at[i, "CoolingE"]=dailyDataFiltered.at[i, "usekWh"]-baseload
                dailyDataFiltered.at[i, "HeatingE"]=0
                dailyDataFiltered.at[i, "BaseE"]=baseload
        else:
            dailyDataFiltered.at[i, "CoolingE"]=0
            dailyDataFiltered.at[i, "HeatingE"]=0
            dailyDataFiltered.at[i, "BaseE"]=dailyDataFiltered.at[i, "usekWh"]
            
            
    #return 값 정의 -2
    yearlyElect=pd.DataFrame({'yearlyCool':[round(dailyDataFiltered.CoolingE.sum(),2).astype('int32')],
                              'yearlyCoolPer':[round(dailyDataFiltered.CoolingE.sum()/dailyDataFiltered.usekWh.sum()*100,1)],
                              'yearlyHeat':[round(dailyDataFiltered.HeatingE.sum(),2).astype('int32')],
                              'yearlyHeatPer':[round(dailyDataFiltered.HeatingE.sum()/dailyDataFiltered.usekWh.sum()*100,1)],
                              'yearlyBase':[round(dailyDataFiltered.BaseE.sum(),2).astype('int32')],
                              'yearlyBasePer':[round(dailyDataFiltered.BaseE.sum()/dailyDataFiltered.usekWh.sum()*100,1)]})
    
    #월별 막대그래프로 그리기 
    MonthlyDataFiltered = round(dailyDataFiltered.resample('1M').sum(),1)
    MonthlyDataFiltered = MonthlyDataFiltered.sort_index()
    MonthlyDataFiltered["monthyear"]=MonthlyDataFiltered.index.astype(str)

    for i in MonthlyDataFiltered.index:
        MonthlyDataFiltered.at[i, "monthyear"]=MonthlyDataFiltered.at[i, "monthyear"].replace('-','')
        MonthlyDataFiltered.at[i, "monthyear"]=MonthlyDataFiltered.at[i, "monthyear"][0:6]

    monthlyElect=MonthlyDataFiltered[["monthyear","usekWh", "HeatingE", "CoolingE", "BaseE"]].astype('int32').reset_index(drop=True)
    
    #계절별 전력량요금 반영해 비용 절감 계산하기 
    sql="""SELECT * FROM ereport.sei_custinfo WHERE "custNo" = '{}'""".format(custNo)
    custinfo=db.execute_query(sql)
        
    if custinfo["selCost"][0]=="0" or 'nan':
        sql="""SELECT * FROM ereport.sei_cost WHERE item = '{}'""".format(custinfo["cntrKnd"][0])
    else:
        sql="""SELECT * FROM ereport.sei_cost WHERE item = '{}' AND sel = '{}'""".format(custinfo["cntrKnd"][0], custinfo["selCost"][0])
    info_costn=db.execute_query(sql)
    
    #saveEfficient : 냉난방기기 효율화로 각각 10% 절감/ 에너지효율화로 기저부하 5% 절감 시 비용/전기사용량 산출 
    saveEfficient=pd.DataFrame({'save' : [save],
                                'costCool':[round(dailyDataFiltered.CoolingE.sum()*0.1*info_costn["summer"].mean())],
                                'elecCool':[round(round(dailyDataFiltered.CoolingE.sum()*0.1))],
                                'costHeat' :[round(dailyDataFiltered.HeatingE.sum()*0.1*info_costn["winter"].mean())],
                                'elecHeat':[round(dailyDataFiltered.HeatingE.sum()*0.1)],
                                'costBase':[round(dailyDataFiltered.BaseE.sum()*0.05*info_costn["sf"].mean())],
                                'elecBase':[round(dailyDataFiltered.BaseE.sum()*0.05)]})
    
    
    #파이그래프 그리기 
    PieE=[dailyDataFiltered.BaseE.sum(),dailyDataFiltered.HeatingE.sum(),dailyDataFiltered.CoolingE.sum()]
    PieColor=["lightgray", "#B32712", "#5C8FE6"]
    fig2=plt.figure()
    plt.pie(PieE, labels=["기저에너지", "난방에너지", "냉방에너지"], colors=PieColor, autopct='%.1f%%')
    plt.savefig('images\\anlypf_2.png', bbox_inches='tight',pad_inches=0.4)
    

    #월별 막대그래프로 그리기 
    fig3 = plt.figure(figsize=(15,5))
    fig3.set_facecolor('white')
    ax = fig3.add_subplot()
    ax.bar(MonthlyDataFiltered["monthyear"], MonthlyDataFiltered["BaseE"], width=0.4, color='lightgray', label='기저부하')
    ax.bar(MonthlyDataFiltered["monthyear"], MonthlyDataFiltered["HeatingE"], bottom=MonthlyDataFiltered["BaseE"], width=0.4, color='#B32712', label='난방부하')
    ax.bar(MonthlyDataFiltered["monthyear"], MonthlyDataFiltered["CoolingE"], bottom=np.add(MonthlyDataFiltered["BaseE"], MonthlyDataFiltered["HeatingE"]), width=0.4, color='#5C8FE6', label='냉방부하')
    current_values = ax.get_yticks()
    ax.set_yticklabels(['{:,.0f}'.format(x) for x in current_values])
    plt.ylabel("월별 전력사용량(kWh)", fontsize=14)
    # plt.title("냉난방효율화 시 예상 절감 비용 : {}만원".format())
    plt.grid(True)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(bbox_to_anchor=(0.7,-0.08), ncol=3, fontsize=14)
    plt.savefig('images\\anlypf_3.png', bbox_inches='tight',pad_inches=0.4)
    
 
    return anlyEfficientGrph, anlyEfficientRgr, Rheat, Rcool, yearlyElect, monthlyElect, saveEfficient, fig1, fig2, fig3, similarity

#### 7_PV분석 
def anlyPV(custNo):
    
    #### db connect ####
    db = DBConnection()
    db.connect()
    
    #건축물대장 정보 불러오기 
    sql="""SELECT * FROM ereport.sei_molitdata WHERE "custNo" = '{}'""".format(custNo)
    data=db.execute_query(sql)
    print("건축물대장 DB 조회 완료, 총 {}동".format(len(data)))
    
    #data2 : 고객정보, 갑/을 구분
    sql="""SELECT * FROM ereport.sei_custinfo WHERE "custNo" = '{}'""".format(custNo)
    data2=db.execute_query(sql)
    
    data["지붕면적"]=data["vlRatEstmTotArea"]/data["grndFlrCnt"]
    data["태양광용량"]=round(data["지붕면적"]/9.9,2)  #3평(9.9m2) 당 1kW
    data["예상발전량"]=round(data["태양광용량"]*3.6*365) #설비용량(kW)*3.6hr(하루 평균 발전시간)*365일
    data["공사비"]=round(data["태양광용량"]*1300000).astype(int)   #설비용량(kW)*130만원(최소기준)
    data["판매형_수익"]=round(data["예상발전량"]*160).astype(int)    #예상발전량(kWh/yr)*전력판매단가(160원/kW)
    data["판매형_ROI"]=round(data["공사비"]/data["판매형_수익"],1) 
    
    
    if '(을)' in data2["cntrKnd"][0]:  #계약종별이 을인 경우, 태양광용량(kW)*3.6hr(하루평균 발전시간)*연간 건물 운영일(주5일 기준)* 발전 시간대 한전 전력 절감 단가 
        data["자가소비형_수익"]=round(data["태양광용량"]*3.6*264*165).astype(int)  #계약전력(을) 기준 고압A 선택II 기준
    else:
        data["자가소비형_수익"]=round(data["태양광용량"]*3.6*264*129).astype(int)  #계약전력(갑) 기준 고압A 선택II 기준 
    
    data["자가소비형_ROI"]=round(data["공사비"]/data["자가소비형_수익"],1)
    
    PVtable1=pd.DataFrame({"PVvol" : [data["태양광용량"].sum()],
                           "preGen" : [data["예상발전량"].sum()],
                           "cost" : [data["공사비"].sum()]})
    
    PVtable2=pd.DataFrame({"selfCost" : [data["자가소비형_수익"].sum()],
                           "selfROI" : [data["자가소비형_ROI"].iloc[0]],
                           "sellCost" : [data["판매형_수익"].sum()],
                           "sellROI" : [data["판매형_ROI"].iloc[0]]})
    
    comment=pd.DataFrame({"meanCost": [(data["자가소비형_수익"].sum()+data["판매형_수익"].sum())/2],
                          "meanROI" : [(data["자가소비형_ROI"].iloc[0]+data["판매형_ROI"].iloc[0])/2]})
    
    return PVtable1, PVtable2, comment