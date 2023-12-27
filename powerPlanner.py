# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 10:48:28 2023

@author: USER
"""


from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.select import Select
from selenium.webdriver.common.keys import Keys
import requests
from bs4 import BeautifulSoup
from html_table_parser import parser_functions
import time
import pandas as pd
import re

# info_costn=pd.read_excel("info_costn_23.xlsx")

class powerPlanner:
    
    def __init__(self, user, password, equipmentID):
        
        self.user = user #고객번호
        self.password = password
        self.equipmentID = equipmentID #계기번호
        
        self.driver = webdriver.Chrome()
        self.url="https://pp.kepco.co.kr/"
        self.driver.get(self.url)
        time.sleep(3)


        #고객번호 입력하기
        self.driver.find_element("xpath", '//*[@id="RSA_USER_ID"]').send_keys(user)
        #비밀번호 입력하기
        self.driver.find_element("xpath", '//*[@id="RSA_USER_PWD"]').send_keys(password)
        #로그인 클릭
        self.driver.find_element("xpath", '//*[@id="intro_form"]/form/fieldset/input[1]').click()
        time.sleep(5)
        print("-----로그인 완료-----")
     

    def getMaxDemand(self, yearDesired):
        #년도 입력 시, 파워플래너/실시간사용량/월별/최대수요 에 있는 표 크롤링 
        
        self.yearDesired = str(yearDesired)
        
        #실시간 사용량 - 월별 진입
        self.urlRealtime = "https://pp.kepco.co.kr/rs/rs0103.do?menu_id=O010203"
        self.driver.get(self.urlRealtime)
        print("-----실시간-월별 사용량 진입-----")
        time.sleep(10)
        
        #최대수요 클릭 
        self.driver.find_element("xpath", '//*[@id="kW"]/a').click()
        time.sleep(1)
        
        #년도 선택 
        selectYear=Select(self.driver.find_element("xpath", '//*[@id="SEARCH_YEAR"]'))
        selectYear.select_by_value(self.yearDesired)

        #조회 버튼 누르기 
        self.driver.find_element("xpath", '//*[@id="txt"]/div[2]/div/p[2]/span[1]/a').click()
        time.sleep(2)
        
        print("-----해당 년도 최대수요 조회-----")

        self.html=self.driver.page_source
        self.response=BeautifulSoup(self.html, 'html.parser')

        data=self.response.find('table', {'id':'grid', 'class':'ui-jqgrid-btable'})
        table=parser_functions.make2d(data)
        self.MaxDemand_raw=pd.DataFrame(data=table[1:], columns=["월", "최대수요(kW)", "전년동월(kW)", "월", "최대수요(kW)", "전년동월(kW)"])
    
        return self.MaxDemand_raw


    def getPowerFactor(self, yearDesired):
        #년도 입력 시, 파워플래너/실시간사용량/월별/역률 에 있는 표 크롤링 
        self.yearDesired = str(yearDesired)
        
        #실시간 사용량 - 월별 진입
        self.urlRealtime = "https://pp.kepco.co.kr/rs/rs0103.do?menu_id=O010203"
        self.driver.get(self.urlRealtime)
        print("-----실시간-월별 사용량 진입-----")
        time.sleep(10)
        
        #역률 클릭 
        self.driver.find_element("xpath", '//*[@id="PF"]/a').click()
        time.sleep(1)
        
        #년도 선택 
        selectYear=Select(self.driver.find_element("xpath", '//*[@id="SEARCH_YEAR"]'))
        selectYear.select_by_value(self.yearDesired)

        #조회 버튼 누르기 
        self.driver.find_element("xpath", '//*[@id="txt"]/div[2]/div/p[2]/span[1]/a').click()
        time.sleep(5)
        
        print("-----해당 년도 역률 조회-----")

        self.html=self.driver.page_source
        self.response=BeautifulSoup(self.html, 'html.parser')

        data=self.response.find('table', {'id':'grid', 'class':'ui-jqgrid-btable'})
        table=parser_functions.make2d(data)
        self.PowerFactor_raw=pd.DataFrame(data=table[1:], columns=["월", "지상역률(%)", "진상역률(%)", "월", "지상역률(%)", "진상역률(%)"])
    
        return self.PowerFactor_raw
    
    
    def getCost(self, yearDesired):
        #년도 입력 시, 파워플래너/요금/월별 청구요금 에 있는 표 크롤링 
        self.yearDesired = str(yearDesired)
        
        #월별 요금
        self.urlRealtime = "https://pp.kepco.co.kr/cc/cc0102.do?menu_id=O010402"
        self.driver.get(self.urlRealtime)
        time.sleep(5)
        print("-----월별 청구요금 진입-----")
   
        #년도 선택 
        selectYear=Select(self.driver.find_element("xpath", '//*[@id="year"]'))
        selectYear.select_by_value(self.yearDesired)

        #조회 버튼 누르기 
        self.driver.find_element("xpath", '//*[@id="txt"]/div[2]/p/span[1]/a').click()
        time.sleep(2)
        
        print("-----해당 년도 요금 조회-----")

        self.html=self.driver.page_source
        self.response=BeautifulSoup(self.html, 'html.parser')

        data=self.response.find('table', {'id':'grid', 'class':'ui-jqgrid-btable'})
        table=parser_functions.make2d(data)
        self.Cost_raw=pd.DataFrame(data=table[1:], columns=["년월", "계약전력(kW)", "요금적용전력(kW)", "사용전력량(kWh)", "사용일수(일)", "요금_지상역률(%)", "요금_진상역률(%)", "전기요금(원)", "기타"])
        self.Cost_raw=self.Cost_raw.drop(["기타"], axis=1)
  
        for i in range(0, len(self.Cost_raw)):
            self.Cost_raw["계약전력(kW)"][i]=self.Cost_raw["계약전력(kW)"][i].replace(',','')
            self.Cost_raw["요금적용전력(kW)"][i]=self.Cost_raw["요금적용전력(kW)"][i].replace(',','')
            self.Cost_raw["사용전력량(kWh)"][i]=self.Cost_raw["사용전력량(kWh)"][i].replace(',','')
            self.Cost_raw["전기요금(원)"][i]=self.Cost_raw["전기요금(원)"][i].replace(',','')  
  
        return self.Cost_raw


    def get15min(self, yearDesired, monthDesired, dayStart, dayEnd):
        #실시간 사용량 진입
        self.urlRealtime = "https://pp.kepco.co.kr/rs/rs0101N.do?menu_id=O010201" 
        self.driver.get(self.urlRealtime)
        time.sleep(5)
        print("-----실시간 사용량 진입-----")
        
        self.yearDesired = str(yearDesired)
        self.monthDesired = str(monthDesired)
        self.dayStart = dayStart
        self.dayEnd = dayEnd

        # Excel template 열기
        self.dataExcel = pd.read_excel("15minDataTemplate3.xlsx", index_col = None)
        
        for i in range(self.dayStart, self.dayEnd+1, 1):
                
            # 캘린더 열기
            calendar = self.driver.find_element("xpath", "//*[@class='ui-datepicker-trigger']")
            calendar.click()
            time.sleep(2)
            
            # 년, 월 선택
            selectYear = Select(self.driver.find_element("xpath", "//*[@class='ui-datepicker-year']"))
            selectYear.select_by_value(self.yearDesired)
            time.sleep(2)
             
            selectMonth = Select(self.driver.find_element("xpath", "//*[@class='ui-datepicker-month']"))
            selectMonth.select_by_visible_text(self.monthDesired)
            time.sleep(2)
            
            # 일 선택
            dateTable = self.driver.find_element("xpath", "//*[@class='ui-datepicker-calendar']/tbody")
            dateRows = dateTable.find_elements(By.TAG_NAME, "tr")
            time.sleep(2)
            
            dayDesired = str(i)           
            daySelected = True
            
            for index, value in enumerate(dateRows):
                dayValue = value.find_elements(By.TAG_NAME, "td")
                
                for w, v in enumerate(dayValue):

                    if v.text == dayDesired:
                        
                        v.click()
                        time.sleep(2)
                        daySelected = False
                        print("-----날짜 선택 완료-----")
                        break
                
                if daySelected == False:
                    
                    # 조회 버튼 클릭        
                    search = self.driver.find_element("xpath", "//*[@class='btn_blue_right']")    
                    search.click()
                    print("-----데이터 조회중-----")
                    time.sleep(5) 
                     
                    # 15분 단위 사용량 데이터 읽어오기
                    tbody = self.driver.find_element("xpath", "//*[@id='tableListChart']/tbody")
                    rows = tbody.find_elements(By.TAG_NAME, "tr")
                    
                    idList = [self.user, self.equipmentID, self.yearDesired + "." + self.monthDesired[:-1] + "." + dayDesired] 
                    consumptionTo12 = [] #오후 12시 데이터 까지 데이터
                    consumptionFrom12 = [] #오후 12시 데이터 부터 데이터
                    
                    for index, value in enumerate(rows):
                        
                        body = value.find_elements(By.TAG_NAME, "td")
                        to12 = body[0]
                        from12 = body[7]
                        
                        consumptionTo12.append(float(to12.text.replace(',','')))
                        consumptionFrom12.append(float(from12.text.replace(',','')))
                        
                    self.consumptionAll = idList + consumptionTo12 + consumptionFrom12 # 사용량 리스트
                    # self.consumptionAll.append(str(sum(float(i.replace(',','')) for i in consumptionTo12) +
                    #                           sum(float(i.replace(',','')) for i in consumptionFrom12))) # 합계 (kWh)
                    self.consumptionAll.append(str(sum(float(i) for i in consumptionTo12) +
                                              sum(float(i) for i in consumptionFrom12))) # 합계 (kWh)
                    
                    self.dataExcel = self.dataExcel.append(pd.Series(self.consumptionAll, index=self.dataExcel.columns), ignore_index=True)
                    print(idList[-1] + '  -----완료-----')
                    
                    break
                          
        return self.dataExcel

    def getPPtotal(self, yearDesired):
        
        self.yearDesired = str(yearDesired)
        self.yearPrevious = str(yearDesired-1)
        
        #실시간 사용량 - 월별 진입
        self.urlRealtime = "https://pp.kepco.co.kr/rs/rs0103.do?menu_id=O010203"
        self.driver.get(self.urlRealtime)
        print("-----실시간-월별 사용량 진입-----")
        time.sleep(15)
        
        #최대수요 클릭 
        self.driver.find_element("xpath", '//*[@id="kW"]/a').click()
        time.sleep(1)
        
        #년도 선택 
        selectYear=Select(self.driver.find_element("xpath", '//*[@id="SEARCH_YEAR"]'))
        selectYear.select_by_value(self.yearDesired)
    
        #조회 버튼 누르기 
        self.driver.find_element("xpath", '//*[@id="txt"]/div[2]/div/p[2]/span[1]/a').click()
        time.sleep(2)
        
        print("-----해당 년도 최대수요 조회-----")
    
        self.html=self.driver.page_source
        self.response=BeautifulSoup(self.html, 'html.parser')
    
        data=self.response.find('table', {'id':'grid', 'class':'ui-jqgrid-btable'})
        table=parser_functions.make2d(data)
        self.MaxDemand_raw=pd.DataFrame(data=table[1:], columns=["년월", "최대수요(kW)", "전년동월(kW)", "월", "최대수요(kW)", "전년동월(kW)"])
        
        self.MaxDemand_edit=pd.DataFrame({'년월' : ['{}12'.format(self.yearDesired), '{}11'.format(self.yearDesired), '{}10'.format(self.yearDesired), '{}09'.format(self.yearDesired), '{}08'.format(self.yearDesired), '{}07'.format(self.yearDesired), '{}06'.format(self.yearDesired), '{}05'.format(self.yearDesired), '{}04'.format(self.yearDesired), '{}03'.format(self.yearDesired), '{}02'.format(self.yearDesired), '{}01'.format(self.yearDesired),
                                                  '{}12'.format(self.yearPrevious), '{}11'.format(self.yearPrevious), '{}10'.format(self.yearPrevious), '{}09'.format(self.yearPrevious), '{}08'.format(self.yearPrevious), '{}07'.format(self.yearPrevious), '{}06'.format(self.yearPrevious), '{}05'.format(self.yearPrevious), '{}04'.format(self.yearPrevious), '{}03'.format(self.yearPrevious), '{}02'.format(self.yearPrevious), '{}01'.format(self.yearPrevious)],
                                          '최대수요(kW)': [self.MaxDemand_raw.iloc[5,4], self.MaxDemand_raw.iloc[4,4],self.MaxDemand_raw.iloc[3,4],self.MaxDemand_raw.iloc[2,4], self.MaxDemand_raw.iloc[1,4], self.MaxDemand_raw.iloc[0,4], self.MaxDemand_raw.iloc[5,1], self.MaxDemand_raw.iloc[4,1],self.MaxDemand_raw.iloc[3,1],self.MaxDemand_raw.iloc[2,1], self.MaxDemand_raw.iloc[1,1], self.MaxDemand_raw.iloc[0,1],
                                                       self.MaxDemand_raw.iloc[5,5],self.MaxDemand_raw.iloc[4,5],self.MaxDemand_raw.iloc[3,5],self.MaxDemand_raw.iloc[2,5],self.MaxDemand_raw.iloc[1,5],self.MaxDemand_raw.iloc[0,5],self.MaxDemand_raw.iloc[5,2],self.MaxDemand_raw.iloc[4,2],self.MaxDemand_raw.iloc[3,2],self.MaxDemand_raw.iloc[2,2],self.MaxDemand_raw.iloc[1,2],self.MaxDemand_raw.iloc[0,2]]})
    
        for i in range(0,len(self.MaxDemand_edit)):
            self.MaxDemand_edit["최대수요(kW)"][i]=self.MaxDemand_edit["최대수요(kW)"][i].replace(',','') 
            
        print("-----최대수요 저장 완료-----")
            
        #역률 클릭 
        time.sleep(1)
        self.driver.find_element("xpath", '//*[@id="PF"]/a').click()
        time.sleep(5)
        
        #년도 선택 
        selectYear=Select(self.driver.find_element("xpath", '//*[@id="SEARCH_YEAR"]'))
        selectYear.select_by_value(self.yearDesired)
    
        #조회 버튼 누르기 
        self.driver.find_element("xpath", '//*[@id="txt"]/div[2]/div/p[2]/span[1]/a/img').click()
        time.sleep(2)
        
        print("-----해당 년도 역률 조회-----")
    
        self.html=self.driver.page_source
        self.response=BeautifulSoup(self.html, 'html.parser')
    
        data=self.response.find('table', {'id':'grid', 'class':'ui-jqgrid-btable'})
        table=parser_functions.make2d(data)
        self.PowerFactor_raw=pd.DataFrame(data=table[1:], columns=["년월", "지상역률(%)", "진상역률(%)", "월", "지상역률(%)", "진상역률(%)"])
        
        #전년도 선택
        selectYear=Select(self.driver.find_element("xpath", '//*[@id="SEARCH_YEAR"]'))
        selectYear.select_by_value(self.yearPrevious)
        
        #조회 버튼 누르기 
        self.driver.find_element("xpath", '//*[@id="txt"]/div[2]/div/p[2]/span[1]/a/img').click()
        time.sleep(5)      
        
        print("-----전년도 역률 조회-----")
        
        self.html=self.driver.page_source
        self.response=BeautifulSoup(self.html, 'html.parser')
    
        data=self.response.find('table', {'id':'grid', 'class':'ui-jqgrid-btable'})
        table=parser_functions.make2d(data)
        self.PowerFactor_pre=pd.DataFrame(data=table[1:], columns=["년월", "지상역률(%)", "진상역률(%)", "월", "지상역률(%)", "진상역률(%)"])

        #당해년도 12월,11월 - 전년도 2월, 1월까지 
        self.PowerFactor_edit=pd.DataFrame({'년월' : ['{}12'.format(self.yearDesired), '{}11'.format(self.yearDesired), '{}10'.format(self.yearDesired), '{}09'.format(self.yearDesired), '{}08'.format(self.yearDesired), '{}07'.format(self.yearDesired), '{}06'.format(self.yearDesired), '{}05'.format(self.yearDesired), '{}04'.format(self.yearDesired), '{}03'.format(self.yearDesired), '{}02'.format(self.yearDesired), '{}01'.format(self.yearDesired),
                                                    '{}12'.format(self.yearPrevious), '{}11'.format(self.yearPrevious), '{}10'.format(self.yearPrevious), '{}09'.format(self.yearPrevious), '{}08'.format(self.yearPrevious), '{}07'.format(self.yearPrevious), '{}06'.format(self.yearPrevious), '{}05'.format(self.yearPrevious), '{}04'.format(self.yearPrevious), '{}03'.format(self.yearPrevious), '{}02'.format(self.yearPrevious), '{}01'.format(self.yearPrevious)],
                                          '지상역률(%)': [self.PowerFactor_raw.iloc[5,4], self.PowerFactor_raw.iloc[4,4],self.PowerFactor_raw.iloc[3,4],self.PowerFactor_raw.iloc[2,4], self.PowerFactor_raw.iloc[1,4], self.PowerFactor_raw.iloc[0,4], self.PowerFactor_raw.iloc[5,1], self.PowerFactor_raw.iloc[4,1],self.PowerFactor_raw.iloc[3,1],self.PowerFactor_raw.iloc[2,1], self.PowerFactor_raw.iloc[1,1], self.PowerFactor_raw.iloc[0,1],
                                                      self.PowerFactor_pre.iloc[5,4], self.PowerFactor_pre.iloc[4,4],self.PowerFactor_pre.iloc[3,4],self.PowerFactor_pre.iloc[2,4], self.PowerFactor_pre.iloc[1,4], self.PowerFactor_pre.iloc[0,4], self.PowerFactor_pre.iloc[5,1], self.PowerFactor_pre.iloc[4,1],self.PowerFactor_pre.iloc[3,1],self.PowerFactor_pre.iloc[2,1], self.PowerFactor_pre.iloc[1,1], self.PowerFactor_pre.iloc[0,1]],
                                          '진상역률(%)' : [self.PowerFactor_raw.iloc[5,5], self.PowerFactor_raw.iloc[4,5],self.PowerFactor_raw.iloc[3,5],self.PowerFactor_raw.iloc[2,5], self.PowerFactor_raw.iloc[1,5], self.PowerFactor_raw.iloc[0,5], self.PowerFactor_raw.iloc[5,2], self.PowerFactor_raw.iloc[4,2],self.PowerFactor_raw.iloc[3,2],self.PowerFactor_raw.iloc[2,2], self.PowerFactor_raw.iloc[1,2], self.PowerFactor_raw.iloc[0,2],
                                                       self.PowerFactor_pre.iloc[5,5], self.PowerFactor_pre.iloc[4,5],self.PowerFactor_pre.iloc[3,5],self.PowerFactor_pre.iloc[2,5], self.PowerFactor_pre.iloc[1,5], self.PowerFactor_pre.iloc[0,5], self.PowerFactor_pre.iloc[5,2], self.PowerFactor_pre.iloc[4,2],self.PowerFactor_pre.iloc[3,2],self.PowerFactor_pre.iloc[2,2], self.PowerFactor_pre.iloc[1,2], self.PowerFactor_pre.iloc[0,2]]})
    
        for i in range(0,len(self.PowerFactor_edit)):
            self.PowerFactor_edit["지상역률(%)"][i]=self.PowerFactor_edit["지상역률(%)"][i].replace(',','') 
            self.PowerFactor_edit["진상역률(%)"][i]=self.PowerFactor_edit["진상역률(%)"][i].replace(',','') 
        
        print("-----역률 저장 완료-----")
        
        #월별 요금
        self.urlRealtime = "https://pp.kepco.co.kr/cc/cc0102.do?menu_id=O010402"
        self.driver.get(self.urlRealtime)
        time.sleep(5)
        print("-----월별 청구요금 진입-----")
    
        #년도 선택 
        selectYear=Select(self.driver.find_element("xpath", '//*[@id="year"]'))
        selectYear.select_by_value(self.yearDesired)
    
        #조회 버튼 누르기 
        self.driver.find_element("xpath", '//*[@id="txt"]/div[2]/p/span[1]/a').click()
        time.sleep(2)
        
        print("-----해당 년도 요금 조회-----")
    
        self.html=self.driver.page_source
        self.response=BeautifulSoup(self.html, 'html.parser')
    
        data=self.response.find('table', {'id':'grid', 'class':'ui-jqgrid-btable'})
        table=parser_functions.make2d(data)
        self.Cost_raw=pd.DataFrame(data=table[1:], columns=["년월", "계약전력(kW)", "요금적용전력(kW)", "사용전력량(kWh)", "사용일수(일)", "요금_지상역률(%)", "요금_진상역률(%)", "전기요금(원)", "기타"])
        self.Cost_raw=self.Cost_raw.drop(["기타"], axis=1)
        
        for i in range(0, len(self.Cost_raw)):
            self.Cost_raw["계약전력(kW)"][i]=self.Cost_raw["계약전력(kW)"][i].replace(',','')
            self.Cost_raw["요금적용전력(kW)"][i]=self.Cost_raw["요금적용전력(kW)"][i].replace(',','')
            self.Cost_raw["사용전력량(kWh)"][i]=self.Cost_raw["사용전력량(kWh)"][i].replace(',','')
            self.Cost_raw["전기요금(원)"][i]=self.Cost_raw["전기요금(원)"][i].replace(',','')
            self.Cost_raw["년월"][i]=self.Cost_raw["년월"][i].replace('년','')
            self.Cost_raw["년월"][i]=self.Cost_raw["년월"][i].replace(' ','')
            self.Cost_raw["년월"][i]=self.Cost_raw["년월"][i].replace('월','')
        
        self.PPtotal=pd.merge(self.Cost_raw, self.MaxDemand_edit, on='년월', how='left')
        self.PPtotal=pd.merge(self.PPtotal, self.PowerFactor_edit, on='년월', how='left')
        # self.PPtotal=pd.concat([self.Cost_raw, self.MaxDemand_edit, self.PowerFactor_edit], axis=1)
        # self.PPtotal=self.PPtotal.drop(["최대수요_월", "역률_월"], axis=1)
        
        # for i in range(0, len(self.PPtotal)):
        #     self.PPtotal["최대수요(kW)"][i]=self.PPtotal["최대수요(kW)"][i].replace('-','')
        #     self.PPtotal["최대수요(kW)"][i]=self.PPtotal["지상역률(%)"][i].replace('-','')
        #     self.PPtotal["최대수요(kW)"][i]=self.PPtotal["진상역률(%)"][i].replace('-','')
        
        # self.PPtotal=self.PPtotal.astype({'년월': 'int', '계약전력(kW)': 'float', "요금적용전력(kW)":"float", "사용전력량(kWh)":"float", "사용일수(일)":"int", "요금_지상역률(%)":"float", "요금_진상역률(%)":"float", "전기요금(원)":"float", 
        #                                   "최대수요(kW)":"float", "지상역률(%)":"float", "진상역률(%)":"float"})
        
        return self.PPtotal