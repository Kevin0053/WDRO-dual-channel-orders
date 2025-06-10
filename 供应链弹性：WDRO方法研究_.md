# **利用分布鲁棒优化解决两渠道多供应商满足订单不确定性以抵抗供应链中断风险**

## **1\. 引言**

### **1.1. 背景与动机：制造与供应链的演变格局**

现代制造业在激烈的全球竞争、经济全球化以及信息和计算机技术的飞速发展下，经历了深刻的变革 [1]。其中一个关键发展是**云制造（Cloud Manufacturing, CMfg）的兴起，这是一种结合了云计算、物联网（IoT）和面向服务技术的新型制造范式** [1]。**云制造旨在解决信息化发展和制造应用中的瓶颈，通过汇集制造资源，实现客户需求与制造商的动态匹配，从而提升敏捷性和响应能力** [1]。与**云制造紧密相关的共享制造（Shared Manufacturing, ShardMfg）**，则是一种按需提供协作生产能力的服务型制造系统 [2]。在过去二十年的经济和社会危机，特别是COVID-19大流行的推动下，共享制造的采纳速度显著加快 [2]。该范式通过点对点协作广泛共享制造能力，旨在提高资源利用率、降低成本并增强供应链韧性[2]。

云制造和共享制造的兴起，不仅仅是技术上的升级，更是对日益加剧的市场波动和经济危机的一种战略性响应。这种范式转变使得制造商能够汇集并动态匹配资源以满足客户需求，从而摆脱了传统的、孤立的制造模式 1。这种固有的灵活性和资源聚合虽然带来了显著优势，但也为管理多样化的需求流和确保运营稳定性带来了新的复杂性。

与此同时，全球供应链正变得越来越容易受到内部和外部中断的影响。这些中断包括经济衰退、重要买家流失、新技术问题、自然灾害、地缘政治紧张以及流行病等 3。例如，COVID-19大流行对全球经济造成了毁灭性影响，凸显了稳健风险管理的迫切需求 4。这些中断可能导致产能短缺、供应延迟、收入减少、市场份额损失和声誉受损 3。此外，“短缺经济”的概念强调了长期资源稀缺性及其相关风险，对传统上假设资源持续可用的供应链规划提出了挑战 5。

供应链中断频率和严重性的增加，加上新兴制造范式（如云制造和共享制造）的出现，共同构成了复杂的运营环境，使得传统的静态供应链规划方法变得不足。在这种波动性下，服务能力不足的问题直接显现，并因云平台的动态特性而加剧，迫切需要一种能够抵御不可预见冲击并有效利用灵活资源的积极适应性能力管理方法。

### **1.2. 问题陈述：双渠道需求与供应能力短缺的管理**

制造商面临着复杂的订单来源环境，包括来自线下传统供应链的随机需求订单和来自云平台的不确定性需求。这种双渠道结构为需求预测和履约带来了显著的复杂性 9。通常，由于成本优势，在线渠道的价格可能较低，这进一步影响了整体的需求动态 10。

此项研究的核心挑战在于**服务能力不足**，这直接导致供应链中断风险 5。能力规划对于满足客户需求和以最优成本运营至关重要 14。能力不足会对供应链绩效产生负面影响 14。

不确定双渠道需求与服务能力不足风险的交汇，构成了一个关键的运营困境。云平台需求的动态性，虽然为资源池化提供了机会，却也加剧了准确预测和匹配供需的挑战，使得制造商在能力管理不当的情况下极易受到中断的影响。这个问题超越了简单的供需匹配，演变为一个多方面的风险管理和韧性建设挑战。

### **1.3. 战略响应：多元化采购与弹性能力**

为了应对外部供应风险和中断，企业正越来越多地采用**多元化采购策略** 16。这种策略通过使用多个供应商来分散中断风险，从而降低对单一来源的依赖 16。它能够增强系统韧性并减轻与单一供应商采购相关的风险 16。

制造商寻求两类供应商来提供相应的服务能力：

* **S1：固定服务能力供应商（Dedicated Capacity Suppliers）**：这些供应商提供预先承诺的固定服务能力水平 17。对于可预测的需求，这提供了稳定性和潜在的较低单位成本。  
* **S2：弹性能力供应商（Elastic Capacity Suppliers），视为“实物期权”**：这些供应商提供灵活的、按需的能力，可以根据实时需求进行调整 18。这种弹性能力被概念化为一种\*\*“实物期权”\*\* \[用户查询\]，允许制造商投资于使用能力的“权利”，而无需预先承诺其全部利用 23。实物期权估值（Real Options Valuation, ROV）是一种先进的分析方法，用于在不确定条件下进行决策，评估延迟决策或改变策略的灵活性价值 23。

灵活调整生产资源对于确保稳健性至关重要 \[用户查询\]。在按订单生产环境中管理弹性能力 19以及制造灵活性对供应链绩效的影响 19都是公认的研究领域。双重采购是创建和利用弹性能力的关键机制 21。

制造商采用固定能力供应商和弹性能力供应商的决策，代表了一种复杂的战略权衡。固定能力为稳定需求提供了可靠的基线和成本效益，而将弹性能力视为“实物期权”，则为应对高度不确定的云平台需求和潜在中断提供了关键的敏捷性和响应能力。这种混合方法旨在平衡效率与韧性，这是现代供应链管理中的一个根本性挑战。实物期权视角为评估这种运营灵活性提供了稳健的理论框架。

### **1.4. 应对不确定性：分布鲁棒优化（WDRO）的作用**

供应链规划常常面临参数被测量误差污染或仅在决策提交后才揭示的情况 25。传统的随机规划假设概率分布精确已知，这在许多实际情境中是不现实的，并可能导致对误设定分布的敏感性 25。鲁棒优化虽然通过基于集合的描述来解决这一问题，但可能过于保守 25。

\*\*分布鲁棒优化（Distributionally Robust Optimization, DRO/WDRO）\*\*是一种先进的方法，用于处理那些不确定参数的精确概率分布本身就不确定的决策问题 25。它融合了随机规划的分布视角和鲁棒优化的最坏情况关注 25。

当仅已知随机数据的部分信息时，例如随机变量的一阶和二阶矩、它们的联合支持或独立性假设时，DRO特别适用 25。这使得在真实分布信息不完全的情况下，能够获得数据驱动的鲁棒解 27。DRO旨在找到在模糊集内最坏分布下表现最优的决策 25。这种最坏情况准则得到了心理学研究的支持，表明许多决策者对分布模糊性容忍度较低 25。它为大规模问题提供了计算上的可处理性，并且在某些情况下可能优于动态规划 30。对于具有长期影响和高固定成本的战略性能力规划决策，当详细的分布信息稀缺时，WDRO是一种特别合适的方法 32。

云平台带来的“不确定性需求”以及能力投资的战略性质，使得WDRO成为一种极其适合的方法。它直接解决了缺乏精确需求分布的实际限制，提供了一种比传统鲁棒优化更稳健但又不过于保守的解决方案。这种方法选择对于确保所提出解决方案的实际适用性和理论严谨性至关重要。

### **1.5. 研究空白与贡献**

尽管云制造和共享制造是新兴范式 1，但现有文献在整体规划和调度共享制造资源方面存在显著空白 2。双渠道供应链研究通常侧重于不确定性下的定价策略和渠道权力 10，但较少关注在云平台需求特定背景下的集成能力和采购决策。供应链风险管理和韧性文献虽然广泛 6，但通常从短期角度考虑中断或假设资源可用性 5。因此，需要能够解决长期资源短缺和级联“涟漪效应”的模型 5。能力管理和采购策略已经探讨了专用和弹性能力 18以及实物期权 23，但在多渠道、云赋能环境中整合这两种不同类型供应商（固定与弹性/实物期权）的研究尚不充分。此外，尽管DRO已应用于供应链管理 28，但其在**双渠道需求不确定性**（尤其是来自云平台）和**供应链中断风险**存在下，联合优化**专用和弹性供应商**之间的能力分配的应用，仍然是一个关键的研究前沿。

本研究通过提出一个新颖的框架来解决上述空白，该框架：

1. **整合云制造动态**：明确建模制造商与云平台的互动，捕捉这种新型制造范式中固有的资源动态池化和不确定性需求。  
2. **管理混合采购策略**：优化订单在两种不同类型供应商之间的分配：传统的固定能力供应商和创新的弹性能力供应商，其中后者使用实物期权方法进行严格的灵活性评估。  
3. **应对双渠道需求不确定性**：开发一个鲁棒决策模型，考虑来自线下渠道的随机需求和来自云平台固有的不确定性需求，这是一种现实且复杂的市场情景。  
4. **通过WDRO增强供应链韧性**：通过采用分布鲁棒优化（WDRO），本研究提供了一个鲁棒解决方案，即使在精确概率分布未知的情况下，也能最小化需求和供应不确定性的最坏情况影响。这直接有助于增强供应链抵抗服务能力不足和中断风险的稳健性。  
5. **提供管理启示**：研究结果将为制造商提供关键的战略和战术指导，帮助他们应对数字化和供应链波动性日益增长的挑战。它将提供关于如何最佳平衡固定和弹性能力、构建供应商合同以及利用云平台以建立能够有效抵抗服务能力不足引起的中断的韧性供应链的见解，最终提升运营绩效和竞争优势。

本研究的核心价值在于其对高度当代问题的整体和集成方法。通过结合前沿的制造范式（云制造）、复杂的采购策略（专用与基于实物期权的弹性能力）、多样的需求结构（双渠道不确定性）和先进的优化技术（WDRO），本研究提供了一个全面的解决方案，超越了现有孤立研究的局限性。这种集成对于解决现代供应链的系统性脆弱性至关重要，并使得本研究在UTD等顶级期刊中具有强大的竞争力。

### **1.6. 论文结构**

本文结构如下：第1节介绍了本研究的背景和动机，概述了问题陈述，引入了战略响应和所选方法，并强调了主要研究空白和贡献。第2节对相关文献进行了全面回顾，涵盖了云制造、双渠道供应链、供应链风险管理、能力采购策略以及不确定性下的优化。第3节详细介绍了模型公式，包括目标函数、约束条件以及WDRO的具体实现。第4节描述了求解方法和计算实验。第5节介绍了数值结果并讨论了关键管理启示。最后，第6节总结了本文并提出了未来的研究方向。

## **2\. 文献综述**

本节对与本研究相关的基础和前沿文献进行了全面回顾，批判性地分析了现有知识，确定了关键理论，并精确指出了本论文旨在填补的研究空白。

### **2.1. 云制造与共享制造范式**

\*\*云制造（CMfg）\*\*被定义为一种新型制造范式，它整合了云计算、物联网（IoT）、面向服务技术和高性能计算等新兴技术 1。其核心目的是克服信息化发展和制造应用中的瓶颈 1。云制造系统的核心组成部分包括云制造资源、制造云服务和制造云本身 1。这种范式通过汇集制造资源，实现了客户需求与可用制造能力的动态匹配，从而显著提升了制造的敏捷性和响应能力 1。

\*\*共享制造（ShardMfg）\*\*是一种服务型制造系统，能够按需提供协作生产能力 2。其发展受到了经济和社会危机的推动，特别是COVID-19大流行加速了其采纳 2。共享制造旨在通过点对点协作广泛共享制造能力，从而降低风险并增强制造供应链的韧性，最终提高资源利用率并通过规模经济降低成本 2。

运营与供应链管理（OSCM）的原则随着这些进步不断演变 34。云制造和共享制造代表了从传统孤立制造向更加互联和面向服务模式的重大转变 2。这种转型影响了OSCM的各个方面，包括能力管理、生产控制和规划 34。

云制造和共享制造的兴起从根本上改变了制造供应链的格局，实现了前所未有的资源池化和动态分配水平。虽然这在通过共享能力提高效率和韧性方面带来了显著优势，但也同时引入了新的复杂性，特别是在管理按需服务固有的不确定性以及这些共享资源的整体规划方面 1。这种转变要求从传统的静态能力规划转向更动态和鲁棒的优化方法。尽管有这些益处，但在这些平台中，高效规划和调度共享制造资源仍存在显著空白 2。在需求不确定性下，将池化的云资源与传统供应链整合的特定挑战仍在探索中 35。

### **2.2. 需求不确定性下的双渠道供应链管理**

现代供应链通常具有多种渠道，例如传统的线下零售和新兴的在线直销平台 10。这种多渠道结构为定价、需求预测和库存管理带来了复杂性 9。

需求不确定性是供应链规划中的一个关键挑战，尤其对于新产品或在快速变化的市场环境中更是如此 9。研究通常利用区间数或其他随机模型来表示不确定的市场需求和价格敏感性 10。文献 10 分析了外部不确定性下双渠道供应链中的定价策略，利用区间数理论和博弈论，重点在于最大化制造商和零售商的预期利润。研究探讨了不同的决策结构（例如，集中式、制造商主导的Stackelberg、零售商主导的Stackelberg、垂直纳什）以及决策者的风险态度（乐观、谨慎、中等）如何影响双渠道设置中的定价决策、市场需求和盈利能力 10。通常，由于成本优势，在线渠道的价格往往较低 10。

当历史数据不可用或需求表现出显著的时空偏差时，准确估计需求变得困难 9。线下和线上需求之间的互动，包括“展厅效应”（showrooming）和“网络效应”（webrooming）等现象，进一步使战略决策复杂化 37。

双渠道需求的存在，特别是引入了动态云平台后，加剧了需求预测和履约中固有的不确定性。这不仅仅是两种需求流的简单叠加效应，而是一种复杂的相互作用，其中一个渠道的特性（例如，云平台需求的不确定性和动态性）会影响另一个渠道以及整个供应链的最佳策略。这种相互依赖性要求采用一种鲁棒的方法，能够处理多样化的需求特征及其潜在的相关性。

### **2.3. 供应链风险管理与韧性**

供应链风险是指可能发生并对供应链绩效产生负面影响的不利和意外事件的可能性 4。供应链中断是扰乱资源正常流动的计划外事件 4。这些中断可以是内部的（例如，经济衰退、买家流失）或外部的（例如，自然灾害、地缘政治紧张、新技术问题、流行病） 3。

中断可能导致显著的业务损失，包括收入减少、供应延迟、市场份额损失和声誉损害 3。供应链中断，特别是在制造业中，一个关键的根本原因是**服务能力不足**和资源稀缺 5。 “短缺经济”的概念强调了这些能力问题的长期系统性，超越了暂时的波动 5。

供应链风险管理（SCRM）是一个系统化、分阶段的方法，用于识别、评估、排序、缓解和监控潜在中断 4。它强调供应链成员之间采取协调一致的方法来降低脆弱性 4。人工智能技术正越来越多地应用于SCRM以增强其能力 39。

供应链韧性（SCR）是指供应链在中断后吸收、适应和恢复自身的能力，以保持运营的连续性 40。它涉及三道防线（能力）：

* **吸收能力（Absorptive Capacity）**：指供应链在中断发生**之前**吸收影响的能力，通过灵活性（生产、采购）、供应商细分、多源策略、不同库存地点和信息共享等策略来实现 41。  
* **适应能力（Adaptive Capacity）**：指供应链在中断**期间**通过实施非标准操作实践来适应和克服中断的能力，通过敏捷性、响应性和效率来实现 41。  
* **恢复能力（Restorative Capacity）**：指当吸收和适应能力不足以维持可接受的绩效水平时，供应链在中断**之后**恢复其流程和运营的能力，通常由技术能力和投资驱动 41。

包括云计算、物联网和大数据分析在内的数字技术，通过支持决策、提供准确数据、改善集成、实现即时响应以及增强主动和被动措施，显著促进了供应链韧性 3。

中断可能在供应链网络中级联传播，这种现象被称为“涟漪效应” 5。当关于中断概率的数据稀缺时，鲁棒优化被应用于分析和缓解涟漪效应的影响 8。

中断频率和严重性的增加，加上“短缺经济”的概念，要求风险管理从被动应对转向主动构建韧性。制造商面临的“服务能力不足”问题，直接与吸收能力和适应能力的建设需求相符。双供应商策略（固定和弹性）是实现多元化和灵活性以构建吸收能力的一种具体体现，而WDRO则提供了分析严谨性，以管理现代中断和涟漪效应所特有的深度不确定性。

**表1：供应链风险类型及缓解策略**

| 风险类别 | 描述/示例 | 关键缓解策略 | 相关文献 |
| :---- | :---- | :---- | :---- |
| 需求风险 | 需求波动、预测不准确、客户行为不可预测 | 需求预测、信息共享、灵活生产、库存池化 | 9 |
| 供应风险 | 供应商故障、单一来源依赖、资源稀缺、质量问题 | 多元化采购、供应商细分、容量预留合同、库存管理 | 4 |
| 运营风险 | 产能短缺、设备故障、劳动力问题、流程中断 | 弹性能力、制造灵活性、冗余能力、应急计划 | 5 |
| 外部/环境风险 | 自然灾害、地缘政治紧张、经济衰退、流行病 | 供应链韧性、风险池化、数字技术、区域供应链 | 3 |
| 财务风险 | 成本波动、汇率变化、供应商财务困境 | 财务对冲、合同设计、垂直整合 | 3 |
| 行为风险 | 决策者偏见、信任/权力问题、信息共享不足 | 行为运营学、激励机制、协作 | 6 |

### **2.4. 能力管理、采购策略与合同**

**战略能力管理**涉及在不确定性下确定能力投资和调整的规模、类型和时机 34。它是有效分配资源以最小化运营成本同时满足需求的关键工具 34。

**专用能力与弹性能力**：

* **专用能力**：指专门分配给特定产品或流程的资源 18。它为可预测的需求提供了效率，但缺乏适应变化的能力 18。  
* **弹性能力**：涉及可以轻松重新配置或扩展/缩减以满足不同需求或应对中断的资源 18。这包括制造灵活性 19以及在按订单生产环境中管理弹性能力的能力 19。虽然弹性策略通常更受欢迎，但如果资源不可靠且企业规避风险，它们可能存在“资源聚合劣势”，这使得选择变得复杂 18。

实物期权理论在能力投资中的应用：  
\*\*实物期权估值（ROV）\*\*已成为在不确定性下进行投资的强大分析方法，它捕捉并评估了许多运营决策中固有的灵活性 23。它允许决策者根据不断变化的市场条件延迟决策或改变策略 23。在供应链管理中，ROV应用于评估合同条款、时机和替代供应链配置的灵活性 24。它解决了供应灵活性（自制/外购、采购）、制造灵活性（流程、生产能力扩张）和分销/物流灵活性（多式联运、仓储能力扩张）等问题 23。  
将弹性能力概念化为“实物期权”是一个至关重要的理论基础。它超越了简单地获取灵活资源，而是评估这些投资中蕴含的**管理灵活性**的价值，尤其是在面对需求和供应不确定性时。这种视角承认动态调整能力的能力具有内在价值，类似于金融期权，可以量化和优化。这对于云平台的不确定性需求尤为重要，因为按需扩展或缩减的能力具有极高价值。

**供应商合同用于能力管理**：

* **能力预留合同**：这些合同允许买方提前向供应商预留一定量的能力 21。研究表明，它们有助于在双向联盟中实现双赢结果，与无合同情景相比，提高了总利润并使双方都受益 44。它们是获得最佳能力配置的一种机制，通常以两部分合同的形式存在 21。  
* **或有合同（Contingent Contracts）**：这些是期权或或有索赔，为买方提供了应对市场变化的灵活性，通常涉及响应型供应商的重新路由或数量灵活性 46。它们可以显著降低成本并提高渠道绩效 46。  
* **双重采购**：一种常见的策略，通过第二个供应来源创建和利用弹性能力 21。它涉及管理两个供应来源，这些来源通常在提前期、能力和成本方面有所不同 21。它连接了混合灵活性和多样化 18。

与供应商（S1和S2）签订不同类型的合同（专用与弹性/基于实物期权）不仅仅是采购行为，更是将韧性嵌入供应链结构的一种方式。能力预留合同和或有合同是运营机制，它们将多元化和灵活采购的战略决策转化为具体的协议，使制造商能够确保能力并适应需求波动和中断。这凸显了合同设计在实现稳健供应链绩效方面的重要性。

**表2：专用能力合同与弹性能力合同比较**

| 合同类型 | 供应商类型 | 关键特征 | 优势 | 挑战/考虑因素 | 相关文献 |
| :---- | :---- | :---- | :---- | :---- | :---- |
| 专用能力合同 | S1 | 固定承诺，预留能力 | 稳定性，稳定需求下的成本效益 | 缺乏灵活性，可能过于保守 | 17 |
| 弹性能力/实物期权合同 | S2 | 按需灵活性，期权预投资，两部分合同 | 敏捷性，对不确定需求的响应性，风险池化，灵活性价值 | 资源聚合劣势（若资源不可靠且规避风险），可变成本较高，估值复杂性 | 18 |

### **2.5. 不确定性下的优化：鲁棒优化与分布鲁棒优化（DRO/WDRO）**

从确定性到随机和鲁棒优化的演变：  
传统上，数学优化问题是确定性的，假设所有数据都已知 25。然而，实际问题中存在不确定性 25。随机规划应运而生，它明确地将不确定参数建模为具有已知概率分布的随机向量，旨在实现最优预期值 25。其局限性包括概率分布精确已知的假设不现实，以及由于“维度诅咒”导致的计算复杂性 25。\*\*鲁棒优化（RO）\*\*则用基于集合的描述取代了概率描述，寻求在“不确定性集合”内最坏情况下表现最佳的解决方案 25。早期的鲁棒优化可能过于保守 26。后来的进展，特别是Bertsimas和Sim（2004）的工作，通过“不确定性预算”参数控制了保守性，同时保持了计算效率 26。  
**分布鲁棒优化（DRO/WDRO）**：

* **定义**：DRO解决的是不确定参数的概率分布本身就不确定的决策问题 25。它旨在找到在“模糊集”内最坏分布下预期值表现最佳的决策 25。  
* **处理部分信息**：当仅已知随机数据的部分信息时，例如随机变量的一阶和二阶矩、它们的联合支持或独立性假设时，DRO特别适用 25。模糊集可以使用φ-散度或Wasserstein距离等差异函数来定义，从而实现可调的保守性 25。  
* **优势**：DRO在传统RO的保守性与随机规划的数据需求之间取得了平衡 25。它提供了对分布误设定不那么敏感的鲁棒解决方案 25。对于大规模问题，它通常在数值上是可处理的 30。

在供应链管理和能力规划中的应用：  
DRO/RO已应用于各种供应链问题，包括需求不确定性下的库存引入 31、具有决策依赖性需求的设施选址问题 28以及随机需求下的通用供应链管理 30。它对于战略能力规划和资源获取决策尤为重要，因为需求预测本身就具有不确定性，并且详细的分布信息稀缺 32。RO/DRO可以深入了解不同程度的鲁棒性如何影响技术的数量、类型和分配 32。当关于中断概率的数据稀缺时，鲁棒优化也用于涟漪效应分析和供应链中断管理以估计风险 8。  
WDRO是本研究的方法论基石，直接解决了问题中的核心不确定性挑战。它能够以最少的分布假设进行操作 30，这使得它对于来自新渠道（如云平台）的精确需求数据可能有限的真实世界供应链具有高度实用性。通过采用WDRO，本研究提供了一种理论上合理且计算上可行的方​​法，以制定对现代商业环境中固有的模糊性具有韧性的战略能力决策，从而弥合了理论模型与实际应用之间的鸿沟。

**表3：不确定性建模方法比较（随机、鲁棒、DRO）**

| 方法 | 对不确定性的信息要求 | 目标 | 保守性水平 | 计算可处理性 | 主要优势 | 主要劣势 | 相关文献 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| 确定性 | 完全确定 | 最小化名义成本 | 无 | 高 | 简单 | 平均值谬误 | 25 |
| 随机规划 | 已知精确概率分布 | 最小化预期成本/风险度量 | 中等 | 通常具有挑战性（维度诅咒） | 对已知分布最优 | 对误设定敏感 | 25 |
| 鲁棒优化 | 基于集合的描述（不确定性集合） | 最小化最坏情况绩效 | 高 | 高（对于线性/锥形问题） | 保证可行性 | 可能过于保守 | 25 |
| 分布鲁棒优化（DRO/WDRO） | 部分概率信息（模糊集，矩） | 最小化最坏情况预期绩效 | 可调/平衡 | 良好（对于某些模糊集） | 数据驱动，比RO更不保守，实用 | 需要部分信息 | 8 |

### **2.6. 综合与研究定位**

前述文献综述揭示了云制造、双渠道供应链、供应链风险管理、弹性能力以及不确定性下先进优化领域日益增长的研究成果。然而，现有研究往往孤立地处理这些领域，或采用简化假设，未能充分捕捉现代制造环境的复杂性。例如，尽管共享制造提高了资源利用率 2，但在高度不确定需求下的整体规划仍然是一个空白 2。双渠道模型通常侧重于定价 10，但很少整合复杂的产能采购决策。供应链韧性文献强调多元化 16和弹性能力 41，但分析工具往往未能处理深度分布模糊性 25。实物期权理论评估灵活性 23，但其与WDRO相结合，用于在双渠道云需求下混合供应商组合的研究尚处于萌芽阶段。

本研究精确地解决了以下研究空白：

1. **云平台动态与双渠道需求的集成建模**：现有文献缺乏一个全面的模型，能够同时考虑来自云制造平台的需求（动态、不确定、资源池化）与传统线下随机需求的独特特征，以及它们对能力规划的综合影响。  
2. **不确定性下的最优混合采购组合**：尽管专用能力和弹性能力被单独研究 18，实物期权理论也应用于能力投资 23，但在应对双渠道需求不确定性和中断风险时，优化与专用供应商和弹性（基于实物期权）供应商**同时**签订合同的**联合战略决策**，存在显著空白。  
3. **WDRO在整体供应链稳健性中的应用**：鲁棒优化在供应链中的现有应用 28通常简化了需求不确定性假设或侧重于特定方面。本研究引入WDRO，通过利用部分分布信息，整体解决需求（来自云平台）和供应（服务能力不足导致的潜在中断）中的深度不确定性，提供了一种更现实和强大的方法来确保供应链的稳健性。  
4. **缓解服务能力不足作为中断的根本原因**：本研究直接解决了“服务能力不足”的关键问题 \[用户查询\]，这是供应链中断的主要驱动因素 5。通过提供一个管理和分配来自不同来源的能力的鲁棒框架，本研究为增强供应链抵抗这一特定紧迫风险的韧性提供了直接解决方案。

本研究的独特贡献：

1. **理论贡献**：本研究通过开发一个新颖的分析框架，协同整合了云制造、双渠道供应链管理、弹性能力的实物期权理论以及分布鲁棒优化等概念，从而推动了运营与供应链管理领域的发展。它为在复杂、模糊的需求和供应不确定性下进行战略采购和能力分配提供了一个严谨的模型，超越了传统的随机或过于保守的鲁棒方法。  
2. **方法论贡献**：本研究展示了WDRO在解决涉及混合供应商合同和新兴平台动态需求的多方面供应链问题中的实际适用性和计算优势。这为管理复杂运营环境中的深度不确定性提供了方法论工具。  
3. **管理贡献**：研究结果为制造商应对数字化和日益增长的供应链波动性挑战提供了关键的战略和战术指导。它提供了关于如何最佳平衡固定和弹性能力、构建供应商合同以及利用云平台以建立能够有效抵抗服务能力不足引起的中断的韧性供应链的见解，最终提升运营绩效和竞争优势。

本研究的独特价值在于其将几个关键但通常独立研究的要素综合整合到一个单一、连贯的框架中。这种整体方法对于解决现代供应链的系统性性质至关重要，特别是新技术（云平台）、不断演变的市场动态（双渠道需求）和持续风险（能力不足）之间的相互作用。通过提供一个鲁棒、数据驱动的决策工具，本研究在实现不确定环境下的主动韧性方面迈出了重要一步，使其与顶级学术期刊高度相关。

**表4：本研究解决的研究空白与贡献**

| 已识别的研究空白 | 本研究的贡献 | 整合的关键概念 |
| :---- | :---- | :---- |
| 缺乏云平台动态与双渠道需求的集成建模 | 开发了一个针对云赋能双渠道供应链的综合模型 | 云制造、双渠道需求、需求不确定性 |
| 优化混合供应商组合的局限性 | 提出了一个优化专用和基于实物期权的弹性采购的框架 | 专用能力、弹性能力、实物期权、供应商合同 |
| WDRO在整体供应链稳健性中应用不足 | 应用WDRO管理深度需求和供应不确定性，以实现整体供应链稳健性 | 分布鲁棒优化（WDRO）、不确定性管理、供应链稳健性 |
| 服务能力不足作为中断根本原因的缓解 | 提供了缓解服务能力不足并增强韧性的鲁棒解决方案 | 服务能力不足、供应链中断风险、供应链韧性 |

#### **引用的著作**

1. (PDF) Cloud manufacturing: A new manufacturing paradigm \- ResearchGate, 访问时间为 六月 10, 2025， [https://www.researchgate.net/publication/241709205\_Cloud\_manufacturing\_A\_new\_manufacturing\_paradigm](https://www.researchgate.net/publication/241709205_Cloud_manufacturing_A_new_manufacturing_paradigm)  
2. Full article: Planning and scheduling shared manufacturing systems ..., 访问时间为 六月 10, 2025， [https://www.tandfonline.com/doi/full/10.1080/00207543.2024.2442549?af=R](https://www.tandfonline.com/doi/full/10.1080/00207543.2024.2442549?af=R)  
3. Supply Chain Resilience and Operational Performance: The Role of ..., 访问时间为 六月 10, 2025， [https://www.mdpi.com/2076-3387/13/2/40](https://www.mdpi.com/2076-3387/13/2/40)  
4. Supply Chain Risk Management: Literature Review \- MDPI, 访问时间为 六月 10, 2025， [https://www.mdpi.com/2227-9091/9/1/16](https://www.mdpi.com/2227-9091/9/1/16)  
5. (PDF) The shortage economy and its implications for supply chain ..., 访问时间为 六月 10, 2025， [https://www.researchgate.net/publication/363497282\_The\_shortage\_economy\_and\_its\_implications\_for\_supply\_chain\_and\_operations\_management](https://www.researchgate.net/publication/363497282_The_shortage_economy_and_its_implications_for_supply_chain_and_operations_management)  
6. A Review of the Existing and Emerging Topics in the Supply Chain ..., 访问时间为 六月 10, 2025， [https://pmc.ncbi.nlm.nih.gov/articles/PMC7283689/](https://pmc.ncbi.nlm.nih.gov/articles/PMC7283689/)  
7. International Journal of Supply and Operations Management May 2022, Volume 9, Issue 2, pp. 162-174 ISSN-Print \- IJSOM, 访问时间为 六月 10, 2025， [http://www.ijsom.com/article\_2871\_f4ce4b7ca7a2e8ccc0a002363207904e.pdf](http://www.ijsom.com/article_2871_f4ce4b7ca7a2e8ccc0a002363207904e.pdf)  
8. (PDF) Ripple effect and supply chain disruption management: new ..., 访问时间为 六月 10, 2025， [https://www.researchgate.net/publication/348570502\_Ripple\_effect\_and\_supply\_chain\_disruption\_management\_new\_trends\_and\_research\_directions](https://www.researchgate.net/publication/348570502_Ripple_effect_and_supply_chain_disruption_management_new_trends_and_research_directions)  
9. Manufacturing & Service Operations Management: Vol 9, No 4 \- PubsOnLine, 访问时间为 六月 10, 2025， [https://pubsonline.informs.org/toc/msom/9/4](https://pubsonline.informs.org/toc/msom/9/4)  
10. Impact of uncertain demand and channel power on dual channel ..., 访问时间为 六月 10, 2025， [https://pmc.ncbi.nlm.nih.gov/articles/PMC10942094/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10942094/)  
11. M\&SOM-Manufacturing & Service Operations Management \- Peeref, 访问时间为 六月 10, 2025， [https://www.peeref.com/journals/5612/m-som-manufacturing-service-operations-management](https://www.peeref.com/journals/5612/m-som-manufacturing-service-operations-management)  
12. Supply Chain Challenges in Competitive World: A Systematic Review and Meta-Analysis of Manufacturing and Service Sectors \- Scientific Research Publishing, 访问时间为 六月 10, 2025， [https://www.scirp.org/journal/paperinformation?paperid=138002](https://www.scirp.org/journal/paperinformation?paperid=138002)  
13. Factors disrupting supply chain management in manufacturing industries \- TU Delft OPEN Journals, 访问时间为 六月 10, 2025， [https://journals.open.tudelft.nl/jscms/article/download/6986/5614/25087](https://journals.open.tudelft.nl/jscms/article/download/6986/5614/25087)  
14. (PDF) THE EFFECT OF CAPACITY PLANNING ON SUPPLY CHAIN PERFORMANCE OF MANUFACTURING FIRMS \- ResearchGate, 访问时间为 六月 10, 2025， [https://www.researchgate.net/publication/388859247\_THE\_EFFECT\_OF\_CAPACITY\_PLANNING\_ON\_SUPPLY\_CHAIN\_PERFORMANCE\_OF\_MANUFACTURING\_FIRMS](https://www.researchgate.net/publication/388859247_THE_EFFECT_OF_CAPACITY_PLANNING_ON_SUPPLY_CHAIN_PERFORMANCE_OF_MANUFACTURING_FIRMS)  
15. The Impact of Supply Chain Complexity on Manufacturing Plant Performance | Request PDF, 访问时间为 六月 10, 2025， [https://www.researchgate.net/publication/222517332\_The\_Impact\_of\_Supply\_Chain\_Complexity\_on\_Manufacturing\_Plant\_Performance](https://www.researchgate.net/publication/222517332_The_Impact_of_Supply_Chain_Complexity_on_Manufacturing_Plant_Performance)  
16. (PDF) Mitigating Supply Chain Risks through Diversified Sourcing Strategies in Manufacturing \- ResearchGate, 访问时间为 六月 10, 2025， [https://www.researchgate.net/publication/387335711\_Mitigating\_Supply\_Chain\_Risks\_through\_Diversified\_Sourcing\_Strategies\_in\_Manufacturing](https://www.researchgate.net/publication/387335711_Mitigating_Supply_Chain_Risks_through_Diversified_Sourcing_Strategies_in_Manufacturing)  
17. Independence of Capacity Ordering and Financial Subsidies to ..., 访问时间为 六月 10, 2025， [https://pubsonline.informs.org/doi/10.1287/msom.1090.0284](https://pubsonline.informs.org/doi/10.1287/msom.1090.0284)  
18. On the Value of Mix Flexibility and Dual Sourcing in Unreliable ..., 访问时间为 六月 10, 2025， [https://pubsonline.informs.org/doi/10.1287/msom.1040.0063](https://pubsonline.informs.org/doi/10.1287/msom.1040.0063)  
19. Publications | Supply Chain Management Research Group, 访问时间为 六月 10, 2025， [https://websites.umass.edu/scmlab/publications/](https://websites.umass.edu/scmlab/publications/)  
20. Operations Management: Processes and Supply Chains, Global Edition, 访问时间为 六月 10, 2025， [http://students.aiu.edu/submissions/profiles/resources/onlineBook/T8x2q2\_Operations%20Management%20Processes%20and%20Supply%20Chains.pdf](http://students.aiu.edu/submissions/profiles/resources/onlineBook/T8x2q2_Operations%20Management%20Processes%20and%20Supply%20Chains.pdf)  
21. Dual sourcing: Creating and utilizing flexible capacities with a ..., 访问时间为 六月 10, 2025， [https://ideas.repec.org/a/bla/popmgt/v31y2022i7p2789-2805.html](https://ideas.repec.org/a/bla/popmgt/v31y2022i7p2789-2805.html)  
22. Automotive Supply Chain Disruption Risk Management: A Visualization Analysis Based on Bibliometric \- MDPI, 访问时间为 六月 10, 2025， [https://www.mdpi.com/2227-9717/11/3/710](https://www.mdpi.com/2227-9717/11/3/710)  
23. Real Options Based Analysis for Supply Chain Management, 访问时间为 六月 10, 2025， [https://ijiepr.iust.ac.ir/article-1-1177-en.pdf](https://ijiepr.iust.ac.ir/article-1-1177-en.pdf)  
24. King's Research Portal \- King's College London Research Portal, 访问时间为 六月 10, 2025， [https://kclpure.kcl.ac.uk/portal/files/82952692/Real\_Options\_in\_Operations\_TRIGEORGIS\_Publishedonline5December2017\_GREEN\_AAM\_CC\_BY\_NC\_ND\_.pdf](https://kclpure.kcl.ac.uk/portal/files/82952692/Real_Options_in_Operations_TRIGEORGIS_Publishedonline5December2017_GREEN_AAM_CC_BY_NC_ND_.pdf)  
25. Distributionally Robust Optimization \- Optimization Online, 访问时间为 六月 10, 2025， [https://optimization-online.org/wp-content/uploads/2024/11/DRO-1.pdf](https://optimization-online.org/wp-content/uploads/2024/11/DRO-1.pdf)  
26. The State of Robust Optimization \- CDN, 访问时间为 六月 10, 2025， [https://bpb-us-w2.wpmucdn.com/people.smu.edu/dist/e/518/files/2017/02/SozuerThiele-final.pdf](https://bpb-us-w2.wpmucdn.com/people.smu.edu/dist/e/518/files/2017/02/SozuerThiele-final.pdf)  
27. Distributionally Robust Stochastic Knapsack Problem | SIAM Journal ..., 访问时间为 六月 10, 2025， [https://epubs.siam.org/doi/10.1137/130915315](https://epubs.siam.org/doi/10.1137/130915315)  
28. Distributionally robust facility location problem under decision-dependent stochastic demand, 访问时间为 六月 10, 2025， [https://par.nsf.gov/servlets/purl/10205721](https://par.nsf.gov/servlets/purl/10205721)  
29. A Distributionally-Robust Service Center Location Problem with Decision Dependent Demand Induced from a Maximum Attraction Principle \- Optimization Online, 访问时间为 六月 10, 2025， [https://optimization-online.org/wp-content/uploads/2020/11/8126.pdf](https://optimization-online.org/wp-content/uploads/2020/11/8126.pdf)  
30. (PDF) A Robust Optimization Approach to Supply Chain Management, 访问时间为 六月 10, 2025， [https://www.researchgate.net/publication/37595220\_A\_Robust\_Optimization\_Approach\_to\_Supply\_Chain\_Management](https://www.researchgate.net/publication/37595220_A_Robust_Optimization_Approach_to_Supply_Chain_Management)  
31. Robust Inventory Induction under Demand Uncertainty \- DSpace@MIT, 访问时间为 六月 10, 2025， [https://dspace.mit.edu/bitstream/handle/1721.1/153669/robin-arobin-sm-orc-2024-thesis.pdf?sequence=1\&isAllowed=y](https://dspace.mit.edu/bitstream/handle/1721.1/153669/robin-arobin-sm-orc-2024-thesis.pdf?sequence=1&isAllowed=y)  
32. Capacity Planning and Resource Acquisition Decisions Using Robust Optimization \- IRL @ UMSL, 访问时间为 六月 10, 2025， [https://irl.umsl.edu/cgi/viewcontent.cgi?article=1178\&context=dissertation](https://irl.umsl.edu/cgi/viewcontent.cgi?article=1178&context=dissertation)  
33. OPERATIONS MANAGEMENT, 访问时间为 六月 10, 2025， [https://ftp.idu.ac.id/wp-content/uploads/ebook/ip/BUKU%20MANAJEMEN%20OPERASI/1390368.pdf](https://ftp.idu.ac.id/wp-content/uploads/ebook/ip/BUKU%20MANAJEMEN%20OPERASI/1390368.pdf)  
34. Operations & Supply Chain Management: Principles and Practice \- arXiv, 访问时间为 六月 10, 2025， [https://arxiv.org/html/2503.05749](https://arxiv.org/html/2503.05749)  
35. Full article: Towards resilient and viable supply chains: a multidimensional model and empirical analysis, 访问时间为 六月 10, 2025， [https://www.tandfonline.com/doi/full/10.1080/00207543.2025.2470350](https://www.tandfonline.com/doi/full/10.1080/00207543.2025.2470350)  
36. The application of digital twin technology in operations and supply chain management: a bibliometric review \- White Rose Research Online, 访问时间为 六月 10, 2025， [https://eprints.whiterose.ac.uk/id/eprint/183445/1/The%20application%20of%20digital%20twin%20technology%20in%20operations%20and%20supply%20chain%20management\_A%20bibliometric%20review.pdf](https://eprints.whiterose.ac.uk/id/eprint/183445/1/The%20application%20of%20digital%20twin%20technology%20in%20operations%20and%20supply%20chain%20management_A%20bibliometric%20review.pdf)  
37. Oben Ceryan \- IDEAS/RePEc, 访问时间为 六月 10, 2025， [https://ideas.repec.org/e/pce136.html](https://ideas.repec.org/e/pce136.html)  
38. Supply Chain Risk Management and Operational Performance: The Enabling Role of Supply Chain Integration \- ResearchGate, 访问时间为 六月 10, 2025， [https://www.researchgate.net/publication/339127225\_Supply\_Chain\_Risk\_Management\_and\_Operational\_Performance\_The\_Enabling\_Role\_of\_Supply\_Chain\_Integration](https://www.researchgate.net/publication/339127225_Supply_Chain_Risk_Management_and_Operational_Performance_The_Enabling_Role_of_Supply_Chain_Integration)  
39. Artificial intelligence applications for supply chain risk management considering interconnectivity, external events exposures and transparency: a systematic literature review | Emerald Insight, 访问时间为 六月 10, 2025， [https://www.emerald.com/insight/content/doi/10.1108/mscra-10-2024-0041/full/html](https://www.emerald.com/insight/content/doi/10.1108/mscra-10-2024-0041/full/html)  
40. Supply Chain Resilience: A Review and Research Direction \- ResearchGate, 访问时间为 六月 10, 2025， [https://www.researchgate.net/publication/357561397\_Supply\_Chain\_Resilience\_A\_Review\_and\_Research\_Direction](https://www.researchgate.net/publication/357561397_Supply_Chain_Resilience_A_Review_and_Research_Direction)  
41. Achieving supply chain resilience in an era of disruptions: a ..., 访问时间为 六月 10, 2025， [https://www.emerald.com/insight/content/doi/10.1108/scm-09-2022-0383/full/html](https://www.emerald.com/insight/content/doi/10.1108/scm-09-2022-0383/full/html)  
42. Operations & Supply Chain Management: Principles and Practice \- arXiv, 访问时间为 六月 10, 2025， [https://arxiv.org/html/2503.05749v1](https://arxiv.org/html/2503.05749v1)  
43. Commissioned Paper: Capacity Management, Investment, and Hedging: Review and Recent Developments \- PubsOnLine, 访问时间为 六月 10, 2025， [https://pubsonline.informs.org/doi/10.1287/msom.5.4.269.24882](https://pubsonline.informs.org/doi/10.1287/msom.5.4.269.24882)  
44. Win-Win Capacity Allocation Contracts in Coproduction and ..., 访问时间为 六月 10, 2025， [https://pubsonline.informs.org/doi/10.1287/mnsc.2015.2358](https://pubsonline.informs.org/doi/10.1287/mnsc.2015.2358)  
45. Comparative Studies of Three Backup Contracts Under Supply Disruptions | Asia-Pacific Journal of Operational Research \- World Scientific Publishing, 访问时间为 六月 10, 2025， [https://www.worldscientific.com/doi/10.1142/S0217595915500062](https://www.worldscientific.com/doi/10.1142/S0217595915500062)  
46. On the Value of Mitigation and Contingency Strategies for Managing Supply Chain Disruption Risks \- PubsOnLine \- Informs.org, 访问时间为 六月 10, 2025， [https://pubsonline.informs.org/doi/10.1287/mnsc.1060.0515](https://pubsonline.informs.org/doi/10.1287/mnsc.1060.0515)  
47. Coordination and Flexibility in Supply Contracts with Options ..., 访问时间为 六月 10, 2025， [https://pubsonline.informs.org/doi/10.1287/msom.4.3.171.7754](https://pubsonline.informs.org/doi/10.1287/msom.4.3.171.7754)  
48. Coordinating Supply and Demand on an On-Demand Service Platform with Impatient Customers | Manufacturing & Service Operations Management \- PubsOnLine, 访问时间为 六月 10, 2025， [https://pubsonline.informs.org/doi/10.1287/msom.2018.0707](https://pubsonline.informs.org/doi/10.1287/msom.2018.0707)  
49. Production & Operations Management \- NIBM E-Library Portal, 访问时间为 六月 10, 2025， [https://nibmehub.com/opac-service/pdf/read/Production%20and%20Operation%20Management.pdf](https://nibmehub.com/opac-service/pdf/read/Production%20and%20Operation%20Management.pdf)  
50. Robust Solutions to Uncertain Semidefinite Programs | SIAM Journal on Optimization, 访问时间为 六月 10, 2025， [https://epubs.siam.org/doi/10.1137/S1052623496305717](https://epubs.siam.org/doi/10.1137/S1052623496305717)