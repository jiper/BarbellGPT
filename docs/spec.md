# 📄 BarbellGPT 项目需求说明书

## 一、项目背景
在力量举训练的实践中，训练者经常会遇到大量需要参考专业知识、训练理论和训练计划设计的问题。这些问题涵盖了从训练计划周期安排、强度调节、疲劳管理、辅助动作选择，到RPE理解、Deload时机判断等内容。
然而，这些知识分散在大量英文教材、训练博客、训练计划模板和论坛对话中，不仅获取门槛高，而且缺乏结构化、可交互的整合方式。
为此，BarbellGPT 致力于打造一个**聚焦力量举领域的智能问答系统**，帮助训练者快速、准确地获取相关知识，并为后续的个性化训练计划设计与训练调控打下基础。
## 二、项目愿景
构建一个“懂你”的力量举训练智能助手，让训练者在面对训练决策时，不再孤立无援，而是拥有一个随时在线、逻辑严谨、可解释、基于一线理论体系和经验总结的 AI 智能伙伴。
## 三、项目目标（阶段性）
### V1阶段：问答系统（RAG问答）
- 提供基于力量举训练理论的问答能力。
- 支持对训练术语、训练理念、计划逻辑等进行解释。
- 提供对典型训练计划的查询和比较。
- 回答具有理论依据、可追溯、语言自然，适应中文或中英混合表达。
### V2阶段：训练计划支持
- 支持通过问答方式获取训练计划模板建议。
- 支持计划结构解读（如体积期、高强度期、Deload结构等）。
- 可基于目标、经验水平、训练频率等因素进行模板筛选。
### V3阶段：训练日志理解与反馈
- 用户可上传日志（如训练记录、HRV、主观评分等）。
- 系统基于日志识别疲劳积累、训练趋势、计划执行偏差。
- 提供针对性建议，如调整强度、替代动作、临时Deload等。
### V4阶段：成为个人智能力量教练
- 在系统理解用户长期训练历史与目标的基础上，自动推荐计划。
- 响应用户每日状态、训练表现，动态调整训练内容。
- 提供训练总结、进步趋势、瓶颈识别与优化建议。
## 四、目标用户
- 有一定训练经验的力量举训练者。
- 希望系统性理解训练理论的进阶者。
- 希望提升训练效率与合理性但缺乏专业指导的训练爱好者。
- 教练辅助工具用户，用于计划构建、知识查找与辅助决策。
## 五、核心价值与功能场景
| 功能场景                  | 用户价值                                      |
|---------------------------|-----------------------------------------------|
| 查询训练术语与概念         | 快速理解RPE、MRV、Deload等理论术语             |
| 比较训练计划结构差异       | 帮助选择合适计划：5/3/1 vs RTS vs 黑又壮等     |
| 分析训练安排是否合理       | 获取疲劳管理建议、辅助动作增删建议等           |
| 制定训练计划               | 推荐适配的模板，并说明设计逻辑                 |
| 分析训练日志反馈           | 识别疲劳累积、训练偏差、目标达成趋势等         |
| 个性化教练支持             | 在时间允许时，持续辅助用户走向训练进步         |
## 六、产品原则
1. **垂直深度优先**：专注于力量举一个领域，做到“说得对、讲得清、回答全”。
2. **理论有据**：所有问答内容均来源于公开出版资料、著名训练体系或教练总结，确保专业性。
3. **可拓展性强**：从问答系统扩展到日志理解、计划自动化、用户建模等智能教练能力。
4. **解释性强**：答案必须给出逻辑解释、出处或模型推理过程，让训练者理解背后原理。
5. **适应性强**：支持中文、中英混合训练术语与问题表达，降低使用门槛。
## 七、里程碑计划
| 阶段    | 时间目标   | 功能亮点                            |
|---------|------------|-------------------------------------|
| V1 MVP  | ~1个月     | 基础RAG问答系统，支持训练理论问答   |
| V2      | ~2-3个月   | 加入训练计划推荐与结构理解能力       |
| V3      | ~4-5个月   | 训练日志输入与反馈建议能力           |
| V4      | 半年+      | 用户建模与持续训练跟踪建议           |
## 八、命名解释
**BarbellGPT**：  
“Barbell”象征训练的核心器械，也代表力量运动的精神；“GPT”代表语言模型能力与生成式AI的基础。  
结合即表示：一个训练领域的智能对话伙伴 —— 你身边的 AI 力量教练。
---
## 九、未来方向（AI力量训练教练）
BarbellGPT 的终极目标不是仅仅成为一个问答机器人，而是能理解你训练路径、历史表现、疲劳状态、心理倾向的私人训练伙伴。它既可以是你训练问题的知识来源，也可以是陪伴你从“新手到大师”的教练型系统。
