class HAction:
    def HInsertFieldTemplate(self, hwp, input, i):
        hwp.HAction.GetDefault("InsertFieldTemplate", hwp.HParameterSet.HInsertFieldTemplate.HSet)
        hwp.HParameterSet.HInsertFieldTemplate.TemplateDirection = input+str(i+1)
        hwp.HParameterSet.HInsertFieldTemplate.TemplateHelp = input+str(i+1)
        hwp.HParameterSet.HInsertFieldTemplate.TemplateName = input+str(i+1)
        hwp.HAction.Execute("InsertFieldTemplate", hwp.HParameterSet.HInsertFieldTemplate.HSet)

    def HFindReplace(self, hwp, input):
        hwp.HAction.GetDefault("RepeatFind", hwp.HParameterSet.HFindReplace.HSet)
        hwp.HParameterSet.HFindReplace.FindString = input
        hwp.HAction.Execute("RepeatFind", hwp.HParameterSet.HFindReplace.HSet)

    def HInsertText(self, hwp, input):
        hwp.HAction.GetDefault("InsertText", hwp.HParameterSet.HInsertText.HSet)
        hwp.HParameterSet.HInsertText.Text = input
        hwp.HAction.Execute("InsertText", hwp.HParameterSet.HInsertText.HSet)