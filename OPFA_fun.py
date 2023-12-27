# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 14:15:49 2022

@author: ninewatt
"""

import bisect

def Find_Goal(hdf, df, GoalPeak, GoalTime):
    #1단계
    Find_Goal1= bisect.bisect_left(hdf["max_elect"], round(max(df["demandkW"]*GoalPeak),-1))
    
    Find_Goal1
    
    #2단계
    Find_Goal2=Find_Goal1+1
    
    for i in range(0,20):
        Find_Goal2=Find_Goal2+1
        if sum(hdf["counts"][Find_Goal2:len(hdf)])<=GoalTime: #피크관리 목표 시간(GoalTime)보다 작은 bin 찾기
            break
        
    Find_Goal2=Find_Goal2-1
    
    Find_Goal2
    
    #3단계
    Find_Goal3=Find_Goal2+1 #Find_Goal3 : 피크관리를 해야 하는 demandkW의 인덱스 
    FinalRange=hdf.loc[Find_Goal3:len(hdf)] #FinalRnage : 피크 관리 해야 하는 기간 
    #Find_Goal3=FinalRange.loc[FinalRange["diff"].idxmax()]["max_elect"]
    Find_Goal3=int(FinalRange[FinalRange["diff"]==FinalRange["diff"].max()].head(n=1)["max_elect"]) #FinalRange 중 변화량(기울기)가 가장 작은 bin의 maxelect값
    
    Final_Goal=Find_Goal3
    
    return(Final_Goal) #Final_Goal : 피크관리 목표 사용량(GoalPeak) 이상 사용하는 기간 중, 피크관리 목표 시간 내에 변화량이 가장 큰 index
    

def Find_Time(hdf, df, GoalPeak, GoalTime):
    #1단계
    Find_Goal1= bisect.bisect_left(hdf["max_elect"], round(max(df["demandkW"]*GoalPeak),-1))
    
    #2단계
    Find_Goal2=Find_Goal1+1
    
    for i in range(0,50):
        Find_Goal2=Find_Goal2+1
        if sum(hdf["counts"][Find_Goal2:len(hdf)])<=GoalTime:
            break
        
    Find_Goal2=Find_Goal2-1
    
    #3단계
    Find_Goal3=Find_Goal2+1
    FinalRange=hdf.loc[Find_Goal3:len(hdf)]
    #Find_Goal3=FinalRange.loc[FinalRange["diff"].idxmax()]["max_elect"]
    Find_Goal3=int(FinalRange[FinalRange["diff"]==FinalRange["diff"].max()].head(n=1)["max_elect"])
    
    Find_Goal3_index=FinalRange[FinalRange["diff"]==FinalRange["diff"].max()].head(n=1).index[0]
    
    FinalRange3=hdf.loc[Find_Goal3_index:len(hdf)]
    Find_Time3=sum(FinalRange3["counts"])
    
    Final_Time=Find_Time3
    
    Final_Time
    
    return(Final_Time)  #Final_Time : 피크관리 목표 사용량(GoalPeak) 이상 사용하는 기간 중, 피크관리 목표 시간 내에 변화량이 가장 큰 index의 시간