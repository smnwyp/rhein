const { defineConfig } = require("cypress");

module.exports = defineConfig({
  e2e: {
    baseUrl: "http://127.0.0.1:8501",
    specPattern: "cypress/e2e/**/*.cy.js",
    supportFile: false,
    video: false,
    viewportWidth: 1440,
    viewportHeight: 1100,
    defaultCommandTimeout: 15000,
  },
});
