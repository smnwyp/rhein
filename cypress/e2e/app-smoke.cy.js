describe("Rhein 核心用户旅程", () => {
  beforeEach(() => cy.visit("/"));

  it("启动后展示五个功能 tab 与数据范围选择器", () => {
    cy.contains("单组回测与 Top 100").should("be.visible");
    cy.contains("参数组合扫描").should("be.visible");
    cy.contains("全部分组组合").should("be.visible");
    cy.contains("分组标的分析").should("be.visible");
    cy.contains("KPI 公式").should("be.visible");
    cy.contains("数据范围").should("be.visible");
  });

  it("可切换到分组，并展示版本选择与条件控件", () => {
    cy.contains("数据范围").parent().click();
    cy.contains("03 高流动性 高波动").click();
    cy.contains("参数预设版本").should("be.visible");
    cy.contains("【T0-01】启用：t0 单日涨幅区间").should("be.visible");
    cy.contains("【EX-04】启用：入场后强制平仓（基准案例）").should("be.visible");
  });
});
