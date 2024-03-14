import sys, os
import psycopg2 as pg
import pandas as pd
import win32com.client as win32
from datetime import datetime
from anlyModule import custInfo, selCost, anlyCtrPeak, anlyPF, anlyEfficient, anlyPV
from h_action import HAction
from db_connect import DBConnection

haction = HAction()
#DB conn & query
db = DBConnection()
db.connect()

# custNm='양주도시공사 에코스포츠센터'
custNo='0322077694'
GoalPeak=0.7
GoalTime=40

#고객번호, 주소-인접 기상청 불러오기 
sql="""SELECT * FROM ereport.sei_address WHERE "custNo" = '{}'""".format(custNo)
address = db.execute_query(sql)
# custNo=address['custNo'][0]
custNm=address['custNm'][0]
pw=address['password'][0]

# custNo='1022760361'
# pw='yjuc9762**'

sql="""SELECT * FROM ereport.sei_kma WHERE "sigunguCd" = '{}'""".format(address['sigunguCd'][0])
kma=db.execute_query(sql)
location=kma['kmaNm'][0]

#15분 단위 데이터 기준으로 분석 기간 선정하기 - 과거데이터 기준
sql="""SELECT * FROM ereport.sei_usekwh WHERE "custNo"= '{}'""".format(custNo)
minute=db.execute_query(sql)
minute.set_index("mrYmd", inplace=True)
minute=minute.sort_index()

start=minute.index[0].date().strftime('%Y-%m-%d')
end=minute.index[len(minute)-1].date().strftime('%Y-%m-%d')

info, Anormaly, usekWhStats, fig1, fig2, fig3, fig4, centerGuide = custInfo(custNo, custNm, start, end)
infoFare, anlySelGraph1, tableData, comment1, anlySelGraph2, comment2, selFare, fig1, fig2 = selCost(custNo, custNm, start, end)
anlyCntrTable, ctrComment, fig1, fig2, anlyPeakTable = anlyCtrPeak(custNo, custNm, start, end, GoalPeak, GoalTime)
anlyPF1, pfComment = anlyPF(custNo, pw, start, end)
anlyEfficientGrph, anlyEfficientRgr, Rheat, Rcool, yearlyElect, monthlyElect, saveEfficient, fig1, fig2, fig3, similarity = anlyEfficient(custNo, start, end, location)
PVtable1, PVtable2, pvComment = anlyPV(custNo)

# hwp 파일 열기
file_root = os.path.abspath(os.path.join(
    os.path.dirname(__file__)
))
file_path = file_root + "/report.hwp"

hwp = win32.gencache.EnsureDispatch("hwpframe.hwpobject")  # 한/글 실행하기
hwp.Open(file_path,"HWP","forceopen:true")
# # hwp.XHwpWindows.Item(0).Visible = True  # 백그라운드 숨김 해제

# print(hwp.GetFieldList())
'''''''''''''''
표지
'''''''''''''''
hwp.PutFieldText("today",datetime.now().strftime('%Y.%m.%d'))
hwp.PutFieldText("custNm",custNm)

'''''''''''''''
고객정보
'''''''''''''''
hwp.PutFieldText("custNo",info['custNo'][0][0:2]+"-"+info['custNo'][0][2:6]+"-"+info['custNo'][0][6:])
hwp.PutFieldText("ikw",format(info['cntrPwr'][0],',d'))
hwp.PutFieldText("ictq",info['cntrKnd'][0]+info['selCost'][0].replace('0', '').replace('1', ' 선택(I)').replace('2', ' 선택(II)').replace('3', ' 선택(III)'))
hwp.PutFieldText("anlyPrd",start +" ~ "+ end)
hwp.PutFieldText("usekWh",'{:,.2f}'.format(info['usekwh'][0]))
hwp.PutFieldText("reqBill",format(info['reqBill'][0],',d'))
hwp.PutFieldText("maxDemand",info['maxDemand'][0])
hwp.PutFieldText("maxDemandTime",info['maxDemandTime'][0])

if info['cntrPwr'][0] > 1000 :
    hwp.PutFieldText("infoComment","계약전력이 1,000kW를 초과하는 건물로써 전기관리자가 상주 및 관리 중일 것으로 예상됩니다. 계약전력 1,000kW 미만 건물 대비 절감 가능성이 낮습니다.")
else :
    hwp.PutFieldText("infoComment","계약전력이 1,000kW를 미만 건물로써 절감 가능성이 높습니다.")

haction.HFindReplace(hwp, "[anoYmd]")
for i in range(len(Anormaly)):
    haction.HInsertFieldTemplate(hwp, "anoYmd", i)
    hwp.HAction.Run("TableRightCellAppend")
    hwp.FindCtrl()
    haction.HInsertFieldTemplate(hwp, "anoUsekwh", i)
    haction.HInsertText(hwp, "kW")
    hwp.HAction.Run("TableAppendRow")
    hwp.HAction.Run("MoveLeft")
    hwp.PutFieldText("anoYmd"+str(i+1),Anormaly['mrYmd'][i].date().strftime('%Y-%m-%d'))
    hwp.PutFieldText("anoUsekwh"+str(i+1),'{:,.2f}'.format(Anormaly['usekWh'][i]))
hwp.HAction.Run("TableSubtractRow")

hwp.PutFieldText("FirstMonth",usekWhStats['FirstMonth'][0])
hwp.PutFieldText("SecondMonth",usekWhStats['SecondMonth'][0])
hwp.PutFieldText("LastMonth",usekWhStats['LastMonth'][0])
hwp.PutFieldText("FirstDay",str(usekWhStats['FirstDay'][0]).replace('0', '월').replace('1', '화').replace('2', '수').replace('3', '목').replace('4', '금').replace('5', '토').replace('6', '일'))
hwp.PutFieldText("SecondDay",str(usekWhStats['SecondDay'][0]).replace('0', '월').replace('1', '화').replace('2', '수').replace('3', '목').replace('4', '금').replace('5', '토').replace('6', '일'))
hwp.PutFieldText("LastDay",str(usekWhStats['LastDay'][0]).replace('0', '월').replace('1', '화').replace('2', '수').replace('3', '목').replace('4', '금').replace('5', '토').replace('6', '일'))

hwp.PutFieldText("lenCenterGuide",len(centerGuide.columns))
haction.HFindReplace(hwp, "[maxIndex]")
for i in range(len(centerGuide.columns)):
    haction.HInsertFieldTemplate(hwp, "maxIndex", i)
    hwp.HAction.Run("TableRightCellAppend")
    hwp.FindCtrl()
    haction.HInsertFieldTemplate(hwp, "maxTime", i)
    haction.HInsertText(hwp, "시")
    hwp.HAction.Run("TableAppendRow")
    hwp.HAction.Run("MoveLeft")
    hwp.PutFieldText("maxIndex"+str(i+1),i+1)
    hwp.PutFieldText("maxTime"+str(i+1),centerGuide[str(i+1)][0])
hwp.HAction.Run("TableSubtractRow")

'''''''''''''''
선택요금제 변경
'''''''''''''''
if tableData.empty == True:
    print("!!!selcost pass!!!")
else:
    haction.HFindReplace(hwp, "[selCost]")
    for i in range(len(tableData)):
        haction.HInsertFieldTemplate(hwp, "selCost", i)
        hwp.HAction.Run("TableRightCellAppend")
        hwp.FindCtrl()
        haction.HInsertFieldTemplate(hwp, "baseBill", i)
        hwp.HAction.Run("TableRightCellAppend")
        hwp.FindCtrl()
        haction.HInsertFieldTemplate(hwp, "kwhBill", i)
        hwp.HAction.Run("TableRightCellAppend")
        hwp.FindCtrl()
        haction.HInsertFieldTemplate(hwp, "envBill", i)
        hwp.HAction.Run("TableRightCellAppend")
        hwp.FindCtrl()
        haction.HInsertFieldTemplate(hwp, "fuelBill", i)
        hwp.HAction.Run("TableRightCellAppend")
        hwp.FindCtrl()
        haction.HInsertFieldTemplate(hwp, "totalBill", i)
        hwp.HAction.Run("TableAppendRow")
        hwp.HAction.Run("MoveUp")
        hwp.HAction.Run("MoveRight")
        hwp.HAction.Run("MoveRight")
        hwp.PutFieldText("selCost"+str(i+1),"선택"+tableData['sel'][i].replace('1', 'I').replace('2', 'II').replace('3', 'III'))
        hwp.PutFieldText("baseBill"+str(i+1),format(tableData['baseBill'][i],',d'))
        hwp.PutFieldText("kwhBill"+str(i+1),format(tableData['kWhBill'][i],',d'))
        hwp.PutFieldText("envBill"+str(i+1),format(tableData['envBill'][i],',d'))
        hwp.PutFieldText("fuelBill"+str(i+1),format(tableData['fuelBill'][i],',d'))
        hwp.PutFieldText("totalBill"+str(i+1),format(tableData['totalBill'][i],',d'))
    hwp.HAction.Run("TableSubtractRow")

    if comment1['saving'][0] == "up":
        hwp.PutFieldText("selComment","기존 “{}” 에서 “{}”로 변경하실 것을 권장하며, 선택요금제 변경을 통해 연간 {:,.0f}원의 비용절감이 예상됩니다. ".format("선택"+comment1['now_plan'][0].replace('1', 'I').replace('2', 'II').replace('3', 'III'),"선택"+comment1['opt_plan'][0].replace('1', 'I').replace('2', 'II').replace('3', 'III'),comment1['save_cost'][0]))
    else:
        hwp.PutFieldText("selComment","현재 요금제 “{}”를 유지하는 것이 가장 적정합니다.".format("선택"+comment1['now_plan'][0].replace('1', 'I').replace('2', 'II').replace('3', 'III')))

    hwp.PutFieldText("selAvgUseTime",comment2['avgUseTime'][0])
    hwp.PutFieldText("selMaxTime",comment2['maxTime'][0])
    hwp.PutFieldText("selMaxMonth",comment2['maxMonth'][0])
    hwp.PutFieldText("selMinTime",comment2['minTime'][0])
    hwp.PutFieldText("selMinMonth",comment2['minMonth'][0])
    
hwp.PutFieldText("selItem",info['cntrKnd'][0]+info['selCost'][0].replace('0', '').replace('1', ' 선택(I)').replace('2', ' 선택(II)').replace('3', ' 선택(III)'))
hwp.PutFieldText("selBase",'{:,.0f}'.format(infoFare['basic'].iloc[0]))

haction.HFindReplace(hwp, "[selLoadNm]")
for i in range(len(infoFare)):
    haction.HInsertFieldTemplate(hwp, "selLoadNm", i)
    hwp.HAction.Run("TableRightCellAppend")
    hwp.FindCtrl()
    haction.HInsertFieldTemplate(hwp, "selSummer", i)
    hwp.HAction.Run("TableRightCellAppend")
    hwp.FindCtrl()
    haction.HInsertFieldTemplate(hwp, "selSf", i)
    hwp.HAction.Run("TableRightCellAppend")
    hwp.FindCtrl()
    haction.HInsertFieldTemplate(hwp, "selWinter", i)
    hwp.HAction.Run("TableAppendRow")
    hwp.HAction.Run("MoveUp")
    hwp.HAction.Run("MoveRight")
    hwp.HAction.Run("MoveRight")
    hwp.HAction.Run("MoveRight")
hwp.HAction.Run("TableSubtractRow")

hwp.HAction.Run("TableColPageUp")
haction.HFindReplace(hwp, "전력량 요금")
hwp.HAction.Run("TableCellBlock")
hwp.HAction.Run("TableCellBlockExtend")
for i in range(len(infoFare)-1):
    hwp.HAction.Run("TableLowerCell")
hwp.HAction.Run("TableMergeCell")
hwp.HAction.Run("Cancel")

if infoFare['loadNm'].iloc[0] == 'nan':
    hwp.HAction.Run("TableCellBlock")
    hwp.HAction.Run("TableCellBlockExtend")
    hwp.HAction.Run("TableRightCell")
    hwp.HAction.Run("TableMergeCell")
    hwp.HAction.Run("SelectAll")
    hwp.HAction.Run("Delete")
    haction.HInsertText(hwp, "전력량 요금")

for i in range(len(infoFare)):
    hwp.PutFieldText("selLoadNm"+str(i+1),infoFare['loadNm'].iloc[i])
    hwp.PutFieldText("selSummer"+str(i+1),'{:,.1f}'.format(infoFare['summer'].iloc[i]))
    hwp.PutFieldText("selSf"+str(i+1),'{:,.1f}'.format(infoFare['sf'].iloc[i]))
    hwp.PutFieldText("selWinter"+str(i+1),'{:,.1f}'.format(infoFare['winter'].iloc[i]))

'''''''''''''''
계약전력 변경
'''''''''''''''
if anlyCntrTable.empty == True:
    print("!!!Cntr pass!!!")
else:
    haction.HFindReplace(hwp, "[billY]")
    hwp.HAction.Run("TableCellBlock")
    hwp.HAction.Run("TableCellBlockExtend")
    hwp.HAction.Run("TableLowerCell")
    hwp.HAction.Run("TableLowerCell")
    hwp.HAction.Run("TableLowerCell")
    hwp.HAction.Run("TableLowerCell")
    hwp.HAction.GetDefault("TableSplitCell", hwp.HParameterSet.HTableSplitCell.HSet)
    hwp.HParameterSet.HTableSplitCell.Rows = 0
    hwp.HParameterSet.HTableSplitCell.Cols = len(anlyCntrTable)
    hwp.HAction.Execute("TableSplitCell", hwp.HParameterSet.HTableSplitCell.HSet)
    hwp.HAction.Run("TableColPageUp")
    hwp.HAction.Run("Cancel")
    haction.HFindReplace(hwp, "[billY]")
    for i in range(len(anlyCntrTable)):
        haction.HInsertFieldTemplate(hwp, "billY", i)
        hwp.HAction.Run("MoveDown")
        haction.HInsertFieldTemplate(hwp, "billYm", i)
        haction.HInsertText(hwp, "월")
        hwp.HAction.Run("MoveDown")
        haction.HInsertFieldTemplate(hwp, "maxPower", i)
        hwp.HAction.Run("MoveDown")
        haction.HInsertFieldTemplate(hwp, "billAplyPwr", i)
        hwp.HAction.Run("MoveDown")
        haction.HInsertFieldTemplate(hwp, "demandkWPC", i)
        hwp.HAction.Run("MoveRight")
        hwp.HAction.Run("MoveUp")
        hwp.HAction.Run("MoveUp")
        hwp.HAction.Run("MoveUp")
        hwp.HAction.Run("MoveUp")
        hwp.PutFieldText("billY"+str(i+1),anlyCntrTable['date'][i][:4])
        hwp.PutFieldText("billYm"+str(i+1),anlyCntrTable['date'][i][4:6])
        hwp.PutFieldText("maxPower"+str(i+1),int(anlyCntrTable['max_power'][i]))
        hwp.PutFieldText("billAplyPwr"+str(i+1),int(anlyCntrTable['billAplyPwr'][i]))
        hwp.PutFieldText("demandkWPC"+str(i+1),anlyCntrTable['demandkWPC'][i])

    hwp.PutFieldText("ctrCntrPwr",format(ctrComment['cntrPwr'][0],',d'))
    hwp.PutFieldText("ctrBillAplyPwrNow",'{:,.0f}'.format(ctrComment['billAplyPwrNow'][0]))
    hwp.PutFieldText("ctrAvgDemandkW",'{:,.0f}'.format(ctrComment['avgDemandkW'][0]))
    hwp.PutFieldText("ctrAvgDemandkWPC",'{:,.1f}'.format(ctrComment['avgDemandkWPC'][0]))
    hwp.PutFieldText("ctrMaxDemandkW",'{:,.0f}'.format(ctrComment['maxDemandkW'][0]))
    hwp.PutFieldText("ctrMaxDemandkWPC",'{:,.1f}'.format(ctrComment['maxDemandkWPC'][0]))

    if comment1['trend'][0] == 'increase': 
        hwp.PutFieldText("ctrTrend","전기사용량은 증가추세이기 때문에,")
    elif comment1['trend'][0] == 'decrease':
        hwp.PutFieldText("ctrTrend","전기사용량은 감소추세이기 때문에,")
    else: 
        hwp.PutFieldText("ctrTrend","전기사용량은 보합추세이기 때문에,")

    if ctrComment['saveCost'][0] == 0:
        hwp.PutFieldText("ctrOptCpower","다음달에")
        hwp.PutFieldText("ctrComment","다시 컨설팅을 받으신 후, 계약전력 변경을 결정하실 것을 권장합니다.")
    else:
        if ctrComment['optCpower'].iloc[0] == ctrComment['cntrPwr'].iloc[0]:
            hwp.PutFieldText("ctrOptCpower","다음달에 다시 컨설팅을 받으신 후, 계약전력 변경을 결정하실 것을 권장합니다.")
            hwp.PutFieldText("ctrComment"," ")
        else:
            hwp.PutFieldText("ctrOptCpower","계약전력을 "+'{:,.0f}'.format(ctrComment['optCpower'][0])+" kW 수준으로 변경을 제안드립니다.")
            hwp.PutFieldText("ctrComment","예상 절감액은 {:,.0f}원입니다. ".format(ctrComment['saveCost'][0]))

'''''''''''''''
피크최적화
'''''''''''''''
if anlyPeakTable.empty == True:
    print("!!!peak pass!!!")
else:
    hwp.PutFieldText("peakCurPeak",'{:,.0f}'.format(anlyPeakTable['curPeak'][0]))
    hwp.PutFieldText("peakOptPeak",'{:,.0f}'.format(anlyPeakTable['optPeak'][0]))
    hwp.PutFieldText("peakManPerc",'{:,.1f}'.format(anlyPeakTable['manPerc'][0]))
    hwp.PutFieldText("peakMngkW",'{:,.0f}'.format(anlyPeakTable['mngkW'][0]))
    hwp.PutFieldText("peakMngTime",'{:,.0f}'.format(anlyPeakTable['mngTime'][0]))
    hwp.PutFieldText("peakExpBill",'{:,.0f}'.format(anlyPeakTable['expBill'][0]))

'''''''''''''''
역률관리
'''''''''''''''
haction.HFindReplace(hwp, "[pfDate]")
for i in range(len(anlyPF1)):
    haction.HInsertFieldTemplate(hwp, "pfDate", i)
    hwp.HAction.Run("TableRightCellAppend")
    hwp.FindCtrl()
    haction.HInsertFieldTemplate(hwp, "pfLaggStan", i)
    hwp.HAction.Run("TableRightCellAppend")
    hwp.FindCtrl()
    haction.HInsertFieldTemplate(hwp, "pfLaggPf", i)
    hwp.HAction.Run("TableRightCellAppend")
    hwp.FindCtrl()
    haction.HInsertFieldTemplate(hwp, "pfLaggPfVar", i)
    hwp.HAction.Run("TableRightCellAppend")
    hwp.FindCtrl()
    haction.HInsertFieldTemplate(hwp, "pfLeadStan", i)
    hwp.HAction.Run("TableRightCellAppend")
    hwp.FindCtrl()
    haction.HInsertFieldTemplate(hwp, "pfLeadPf", i)
    hwp.HAction.Run("TableRightCellAppend")
    hwp.FindCtrl()
    haction.HInsertFieldTemplate(hwp, "pfLeadPfVar", i)
    hwp.HAction.Run("TableRightCellAppend")
    hwp.FindCtrl()
    haction.HInsertFieldTemplate(hwp, "pfBill", i)
    hwp.HAction.Run("TableAppendRow")
    hwp.HAction.Run("MoveUp")
    hwp.HAction.Run("MoveRight")
    hwp.HAction.Run("MoveRight")
    hwp.PutFieldText("pfDate"+str(i+1),anlyPF1['date'][i])
    hwp.PutFieldText("pfLaggStan"+str(i+1),anlyPF1['laggStan'][i])
    hwp.PutFieldText("pfLaggPf"+str(i+1),'{:,.2f}'.format(anlyPF1['laggPF'][i]))
    hwp.PutFieldText("pfLaggPfVar"+str(i+1),'{:,.2f}'.format(anlyPF1['laggPFVar'][i]))
    hwp.PutFieldText("pfLeadStan"+str(i+1),anlyPF1['leadStan'][i])
    hwp.PutFieldText("pfLeadPf"+str(i+1),'{:,.2f}'.format(anlyPF1['leadPF'][i]))
    hwp.PutFieldText("pfLeadPfVar"+str(i+1),'{:,.2f}'.format(anlyPF1['leadPFVar'][i]))
    hwp.PutFieldText("pfBill"+str(i+1),'{:,.0f}'.format(anlyPF1['pfBill'][i]))
hwp.HAction.Run("TableSubtractRow")

pfBill = 0
saveSign = ''
if pfComment['saveCost'][0] < 0: 
    saveSign='-'
    pfBill = abs(pfComment['saveCost'][0])
else: 
    saveSign='+'
    pfBill = 0

if pfComment['laggChange'][0] == 1 and pfComment['leadChange'][0] == 1:
    hwp.PutFieldText("pfComment1","지상역률, 진상역률이 모두 기준역률(지상역률 90%, 진상역률 95%) 미만으로")
    hwp.PutFieldText("pfComment2","분석기간 동안 {:,.0f}원의 요금이 {}되었습니다. 역률요금 할인을 위하여 지상역률과 진상역률 관리를 권장합니다.".format(abs(pfComment['saveCost'][0]),saveSign.replace('+', '추가로 부과').replace('-', '감액')))
elif pfComment['laggChange'][0] == 0 and pfComment['leadChange'][0] == 1:
    hwp.PutFieldText("pfComment1","지상역률은 기준역률(지상역률 90%) 이상이나, 진상역률은 기준역률(진상역률 95%) 미만으로")
    hwp.PutFieldText("pfComment2","분석기간 동안 {:,.0f}원의 요금이 {}되었습니다. 역률요금 할인을 위하여 진상역률 관리를 권장합니다.".format(abs(pfComment['saveCost'][0]),saveSign.replace('+', '추가로 부과').replace('-', '감액')))
elif pfComment['laggChange'][0] == 1 and pfComment['leadChange'][0] == 0:
    hwp.PutFieldText("pfComment1","진상역률은 기준역률(진상역률 95%) 이상이나, 지상역률은 기준역률(지상역률 90%) 미만으로")        
    hwp.PutFieldText("pfComment2","분석기간 동안 {:,.0f}원의 요금이 {}되었습니다. 역률요금 할인을 위하여 지상역률 관리를 권장합니다.".format(abs(pfComment['saveCost'][0]),saveSign.replace('+', '추가로 부과').replace('-', '감액')))
else :
    hwp.PutFieldText("pfComment1","지상역률, 진상역률이 모두 기준역률(지상역률 90%, 진상역률 95%) 이상으로")
    hwp.PutFieldText("pfComment2","분석기간 동안 {:,.0f}원의 요금이 감액되었습니다. 현재 수준의 역률을 유지하기 위한 지속적인 관리를 권장합니다.".format(abs(pfComment['saveCost'][0])))

'''''''''''''''
설비효율화
'''''''''''''''
if similarity == '0' :
    hwp.PutFieldText("similarity","유의미하게 다릅니다.")
else:
    hwp.PutFieldText("similarity","유사합니다.")

hwp.PutFieldText("Rheat",Rheat)
hwp.PutFieldText("Rcool",Rcool)
if Rheat < -0.6: #높은 난방 민감도
    hwp.PutFieldText("RheatComment","높은 난방 민감도를 보입니다.")
elif Rheat < -0.2 and Rheat > -0.6: #보통 난방 민감도
    hwp.PutFieldText("RheatComment","보통 난방 민감도를 보입니다.")
elif Rheat < 0 and Rheat > -0.2: #낮은 난방 민감도
    hwp.PutFieldText("RheatComment","낮은 난방 민감도를 보입니다.")
else:
    hwp.PutFieldText("RheatComment","비정상적인 난방 가동 또는 난방설비 미사용으로 보입니다.")

if Rcool > 0.6: #높은 냉방 민감도
    hwp.PutFieldText("RcoolComment","높은 냉방 민감도를 보입니다.")
elif Rcool > 0.2 and Rcool < 0.6: #보통 냉방 민감도
    hwp.PutFieldText("RcoolComment","보통 냉방 민감도를 보입니다.")
elif Rcool > 0 and Rcool < 0.2: #낮은 냉방 민감도
    hwp.PutFieldText("RcoolComment","낮은 냉방 민감도를 보입니다.")
else:
    hwp.PutFieldText("RcoolComment","비정상적인 냉방 가동 또는 냉방설비 미사용으로 보입니다.")


if Rheat <= -0.6: #높은 난방 민감도
    if Rcool >= 0.6:
        hwp.PutFieldText("efComment1","냉난방 설비 효율화를 통한 에너지 절감 가능성이 높습니다.")
    else :
        hwp.PutFieldText("efComment1","난방 설비 효율화를 통한 에너지 절감 가능성은 높으며, 냉방 설비 효율화를 통한 에너지 절감 가능성이 낮습니다.")
elif Rheat <= -0.2 and Rheat > -0.6: #보통 난방 민감도
    if Rcool >= 0.6:
        hwp.PutFieldText("efComment1","난방 설비 효율화를 통한 에너지 절감 가능성은 낮으며, 냉방 설비 효율화를 통한 에너지 절감 가능성이 높습니다.")
    else:
        hwp.PutFieldText("efComment1","난방 설비 효율화를 통한 에너지 절감 가능성이 있으며, 냉방 설비 효율화를 통한 에너지 절감 가능성이 낮습니다.")
elif Rheat < 0 and Rheat > -0.2: #낮은 난방 민감도
    if Rcool > 0:
        hwp.PutFieldText("efComment1","난방 설비 효율화를 통한 에너지 절감 가능성은 낮으며, 냉방 설비 효율화를 통한 에너지 절감 가능성이 있습니다.")
    else:
        hwp.PutFieldText("efComment1","냉난방 설비 효율화를 통한 에너지 절감 가능성이 낮습니다.")
else:
    if Rcool > 0:
        hwp.PutFieldText("efComment1","난방 설비 효율화를 통한 에너지 절감 가능성은 낮으며, 냉방 설비 효율화를 통한 에너지 절감 가능성이 있습니다.")
    else:
        hwp.PutFieldText("efComment1","냉난방 설비 효율화를 통한 에너지 절감 가능성이 낮습니다.")


hwp.PutFieldText("yearlyBase",'{:,.2f}'.format(yearlyElect['yearlyBase'][0]))
hwp.PutFieldText("yearlyBasePer",'{:,.1f}'.format(yearlyElect['yearlyBasePer'][0]))
hwp.PutFieldText("yearlyHeat",'{:,.2f}'.format(yearlyElect['yearlyHeat'][0]))
hwp.PutFieldText("yearlyHeatPer",'{:,.1f}'.format(yearlyElect['yearlyHeatPer'][0]))
hwp.PutFieldText("yearlyCool",'{:,.2f}'.format(yearlyElect['yearlyCool'][0]))
hwp.PutFieldText("yearlyCoolPer",'{:,.1f}'.format(yearlyElect['yearlyCoolPer'][0]))

hwp.PutFieldText("costBase",'{:,.0f}'.format(saveEfficient['costBase'][0]))
hwp.PutFieldText("elecBase",'{:,.1f}'.format(saveEfficient['elecBase'][0]))

if saveEfficient['save'][0] == 0:
    hwp.PutFieldText("efComment2"," ")
    efTotalBill = 0
elif saveEfficient['save'][0] == 1:
    hwp.PutFieldText("efComment2","히트펌프, 난방실외기 등 기기 효율화로 난방부하 10% 절감 시, 연간 {:,.0f}원({:,.2f}kWh) 비용 절감이 예상됩니다.".format(saveEfficient['costHeat'][0],saveEfficient['elecHeat'][0]))
    efTotalBill = saveEfficient['costBase'][0]+saveEfficient['costHeat'][0]
elif saveEfficient['save'][0] == 2:
    hwp.PutFieldText("efComment2","히트펌프, 냉방실외기 등 기기 효율화로 냉방부하 10% 절감 시, 연간 {:,.0f}원({:,.2f}kWh) 비용 절감이 예상됩니다.".format(saveEfficient['costCool'][0],saveEfficient['elecCool'][0]))
    efTotalBill = saveEfficient['costBase'][0]+saveEfficient['costCool'][0]
else :
    hwp.PutFieldText("efComment2","히트펌프, 냉난방실외기 등 기기 효율화로 냉방부하 10% 절감 시, 연간 {:,.0f}원({:,.2f}kWh), 난방부하 10% 절감 시, 연간 연간 {:,.0f}원({:,.2f}kWh) 비용 절감이 예상됩니다.".format(saveEfficient['costCool'][0],saveEfficient['elecCool'][0],saveEfficient['costHeat'][0],saveEfficient['elecHeat'][0]))
    efTotalBill = saveEfficient['costBase'][0]+saveEfficient['costHeat'][0]+saveEfficient['costCool'][0]

'''''''''''''''
신재생에너지
'''''''''''''''
hwp.PutFieldText("PVvol",'{:,.2f}'.format(PVtable1['PVvol'][0]))
hwp.PutFieldText("preGen",'{:,.0f}'.format(PVtable1['preGen'][0]))
hwp.PutFieldText("pvCost",'{:,.0f}'.format(PVtable1['cost'][0]))
hwp.PutFieldText("selfCost",'{:,.0f}'.format(PVtable2['selfCost'][0]))
hwp.PutFieldText("selfROI",'{:,.1f}'.format(PVtable2['selfROI'][0]))
hwp.PutFieldText("sellCost",'{:,.0f}'.format(PVtable2['sellCost'][0]))
hwp.PutFieldText("sellROI",'{:,.1f}'.format(PVtable2['sellROI'][0]))

hwp.PutFieldText("meanCost",'{:,.0f}'.format(pvComment['meanCost'][0]))
hwp.PutFieldText("meanROI",'{:,.1f}'.format(pvComment['meanROI'][0]))
# hwp.PutFieldText("custAddress",custAddress)

'''''''''''''''
summary
'''''''''''''''
if tableData.empty == True:
    hwp.PutFieldText("selSaveCost",'{:,.0f}'.format(0))
    selSaveCost=0
else:
    hwp.PutFieldText("selSaveCost",'{:,.0f}'.format(comment1['save_cost'][0]))
    selSaveCost=comment1['save_cost'][0]
if anlyCntrTable.empty == True:
    hwp.PutFieldText("ctrSaveCost",'{:,.0f}'.format(0))
    ctrSaveCost=0
else:
    hwp.PutFieldText("ctrSaveCost",'{:,.0f}'.format(ctrComment['saveCost'][0]))
    ctrSaveCost=ctrComment['saveCost'][0]
if anlyPeakTable.empty == True:
    hwp.PutFieldText("peakExpBill",'{:,.0f}'.format(0))
    peakExpBill=0
else:
    hwp.PutFieldText("peakExpBill",'{:,.0f}'.format(anlyPeakTable['expBill'][0]))
    peakExpBill=anlyPeakTable['expBill'][0]
hwp.PutFieldText("pfBill",'{:,.0f}'.format(pfBill))
hwp.PutFieldText("efTotalBill",'{:,.0f}'.format(efTotalBill))
hwp.PutFieldText("meanCost",'{:,.0f}'.format(pvComment['meanCost'][0]))
hwp.PutFieldText("totalBill",'{:,.0f}'.format(selSaveCost+ctrSaveCost+peakExpBill+pfComment['saveCost'][0]+efTotalBill+pvComment['meanCost'][0]))

image_path = file_root + "/images/"
for file in os.listdir(image_path):
    img_path = file_root + "/images/"+ file
    print(img_path)

    while True:

        # 이미지 삽입할 위치 찾기
        hwp.HAction.GetDefault("RepeatFind", hwp.HParameterSet.HFindReplace.HSet)
        hwp.HParameterSet.HFindReplace.FindString = "["+file+"]"
        hwp.HParameterSet.HFindReplace.IgnoreMessage = 1
        result = hwp.HAction.Execute("RepeatFind", hwp.HParameterSet.HFindReplace.HSet)
        # hwp.MovePos()
        print("위치 찾기=>", result)

        # 다 바꿨으면 종료
        if result == False:
            break

        hwp.InsertPicture(img_path, Embedded=True, sizeoption=3) # 이미지 삽입
        hwp.FindCtrl() # 이미지 선택
        hwp.HAction.Run("Cut") # 잘라내기 (커서에서 인접한 개체 선택)
        # 이미지 붙여넣기
        hwp.HAction.GetDefault("Paste", hwp.HParameterSet.HSelectionOpt.HSet)
        result = hwp.HAction.Execute("Paste", hwp.HParameterSet.HSelectionOpt.HSet)
        print("붙여넣기=>", result)

hwp.SaveAs(os.getcwd()+"\\"+custNm+".hwp")

def DeleteAllFiles(filePath):
    if os.path.exists(filePath):
        for file in os.scandir(filePath):
            os.remove(file.path)
        return print('Remove All File')
    else:
        return print('Directory Not Found')

DeleteAllFiles(os.getcwd()+"\\images")
