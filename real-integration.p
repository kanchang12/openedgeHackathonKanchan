/* 
   Final ABL Script for ML Price Optimization
   Works with API on http://localhost:5001/
*/

DEFINE VARIABLE cProductJson    AS CHARACTER NO-UNDO.
DEFINE VARIABLE cResponse       AS CHARACTER NO-UNDO.
DEFINE VARIABLE cStockCode      AS CHARACTER NO-UNDO.
DEFINE VARIABLE cDescription    AS CHARACTER NO-UNDO.
DEFINE VARIABLE dCurrentPrice   AS DECIMAL   NO-UNDO.
DEFINE VARIABLE iQuantity       AS INTEGER   NO-UNDO.
DEFINE VARIABLE dOptimalPrice   AS DECIMAL   NO-UNDO.
DEFINE VARIABLE dCurrentRevenue AS DECIMAL   NO-UNDO.
DEFINE VARIABLE dOptimalRevenue AS DECIMAL   NO-UNDO.
DEFINE VARIABLE cElasticityType AS CHARACTER NO-UNDO.
DEFINE VARIABLE iPos            AS INTEGER   NO-UNDO.
DEFINE VARIABLE iEnd            AS INTEGER   NO-UNDO.

/* Open log file */
OUTPUT TO "ml_session.log".

PUT UNFORMATTED "================================================" SKIP.
PUT UNFORMATTED "        ML PRICE OPTIMIZATION SESSION" SKIP.
PUT UNFORMATTED "        Training: 379,336 | Test: 108,436" SKIP.
PUT UNFORMATTED "        R² Score: 0.291" SKIP.
PUT UNFORMATTED "================================================" SKIP(1).

/* Step 1: Get Random Product */
PUT UNFORMATTED "Getting random product from API..." SKIP.

OS-COMMAND SILENT 'curl -s "http://localhost:5001/api/product/random" > product.json'.

INPUT FROM "product.json" NO-ECHO.
cProductJson = "".
IMPORT UNFORMATTED cProductJson.
INPUT CLOSE.

IF LENGTH(cProductJson) = 0 THEN DO:
    PUT UNFORMATTED "ERROR: No response from API" SKIP.
    OUTPUT CLOSE.
    MESSAGE "Error: API not responding. Check if server is running on port 5001" 
        VIEW-AS ALERT-BOX ERROR.
    RETURN.
END.

/* Parse Product JSON */
/* Extract stock_code */
iPos = INDEX(cProductJson, '"stock_code"').
IF iPos > 0 THEN DO:
    iPos = INDEX(cProductJson, ':', iPos) + 1.
    /* Skip whitespace and quotes */
    DO WHILE SUBSTRING(cProductJson, iPos, 1) = ' ' OR 
             SUBSTRING(cProductJson, iPos, 1) = '"':
        iPos = iPos + 1.
    END.
    iEnd = INDEX(cProductJson, '"', iPos).
    cStockCode = SUBSTRING(cProductJson, iPos, iEnd - iPos).
END.

/* Extract description */
iPos = INDEX(cProductJson, '"description"').
IF iPos > 0 THEN DO:
    iPos = INDEX(cProductJson, ':', iPos) + 1.
    DO WHILE SUBSTRING(cProductJson, iPos, 1) = ' ' OR 
             SUBSTRING(cProductJson, iPos, 1) = '"':
        iPos = iPos + 1.
    END.
    iEnd = INDEX(cProductJson, '"', iPos).
    cDescription = SUBSTRING(cProductJson, iPos, iEnd - iPos).
END.

/* Extract current_price */
iPos = INDEX(cProductJson, '"current_price"').
IF iPos > 0 THEN DO:
    iPos = INDEX(cProductJson, ':', iPos) + 1.
    DO WHILE SUBSTRING(cProductJson, iPos, 1) = ' ':
        iPos = iPos + 1.
    END.
    iEnd = INDEX(cProductJson, ',', iPos).
    IF iEnd = 0 THEN iEnd = INDEX(cProductJson, '}', iPos).
    dCurrentPrice = DECIMAL(TRIM(SUBSTRING(cProductJson, iPos, iEnd - iPos))) NO-ERROR.
END.

/* Extract quantity */
iPos = INDEX(cProductJson, '"quantity"').
IF iPos > 0 THEN DO:
    iPos = INDEX(cProductJson, ':', iPos) + 1.
    DO WHILE SUBSTRING(cProductJson, iPos, 1) = ' ':
        iPos = iPos + 1.
    END.
    iEnd = INDEX(cProductJson, ',', iPos).
    IF iEnd = 0 THEN iEnd = INDEX(cProductJson, '}', iPos).
    iQuantity = INTEGER(TRIM(SUBSTRING(cProductJson, iPos, iEnd - iPos))) NO-ERROR.
END.

PUT UNFORMATTED "Product fetched successfully:" SKIP.
PUT UNFORMATTED "  Stock Code:  " cStockCode SKIP.
PUT UNFORMATTED "  Description: " cDescription SKIP.
PUT UNFORMATTED "  Price:       £" STRING(dCurrentPrice, ">>9.99") SKIP.
PUT UNFORMATTED "  Quantity:    " iQuantity SKIP(1).

/* Step 2: Get ML Prediction */
PUT UNFORMATTED "Requesting ML prediction..." SKIP.

/* Write product JSON to file for POST */
OUTPUT TO "request.json".
PUT UNFORMATTED cProductJson.
OUTPUT CLOSE.

OUTPUT TO "ml_session.log" APPEND.

/* Call prediction API */
OS-COMMAND SILENT 'curl -s -X POST "http://localhost:5001/api/predict/elasticity-curve" -H "Content-Type: application/json" -d @request.json > prediction.json'.

INPUT FROM "prediction.json" NO-ECHO.
cResponse = "".
IMPORT UNFORMATTED cResponse.
INPUT CLOSE.

IF INDEX(cResponse, '"error"') > 0 THEN DO:
    PUT UNFORMATTED "ERROR: Prediction failed" SKIP.
    OUTPUT CLOSE.
    MESSAGE "Error: ML prediction failed" VIEW-AS ALERT-BOX ERROR.
    OS-DELETE VALUE("product.json").
    OS-DELETE VALUE("request.json").
    OS-DELETE VALUE("prediction.json").
    RETURN.
END.

/* Parse Prediction Response */
/* Extract optimal_price */
iPos = INDEX(cResponse, '"optimal_price"').
IF iPos > 0 THEN DO:
    iPos = INDEX(cResponse, ':', iPos) + 1.
    DO WHILE SUBSTRING(cResponse, iPos, 1) = ' ':
        iPos = iPos + 1.
    END.
    iEnd = INDEX(cResponse, ',', iPos).
    IF iEnd = 0 THEN iEnd = INDEX(cResponse, '}', iPos).
    dOptimalPrice = DECIMAL(TRIM(SUBSTRING(cResponse, iPos, iEnd - iPos))) NO-ERROR.
END.

/* Extract current_revenue */
iPos = INDEX(cResponse, '"current_revenue"').
IF iPos > 0 THEN DO:
    iPos = INDEX(cResponse, ':', iPos) + 1.
    DO WHILE SUBSTRING(cResponse, iPos, 1) = ' ':
        iPos = iPos + 1.
    END.
    iEnd = INDEX(cResponse, ',', iPos).
    IF iEnd = 0 THEN iEnd = INDEX(cResponse, '}', iPos).
    dCurrentRevenue = DECIMAL(TRIM(SUBSTRING(cResponse, iPos, iEnd - iPos))) NO-ERROR.
END.

/* Extract optimal_revenue */
iPos = INDEX(cResponse, '"optimal_revenue"').
IF iPos > 0 THEN DO:
    iPos = INDEX(cResponse, ':', iPos) + 1.
    DO WHILE SUBSTRING(cResponse, iPos, 1) = ' ':
        iPos = iPos + 1.
    END.
    iEnd = INDEX(cResponse, ',', iPos).
    IF iEnd = 0 THEN iEnd = INDEX(cResponse, '}', iPos).
    dOptimalRevenue = DECIMAL(TRIM(SUBSTRING(cResponse, iPos, iEnd - iPos))) NO-ERROR.
END.

/* Extract elasticity_type */
cElasticityType = "".
iPos = INDEX(cResponse, '"elasticity_type"').
IF iPos > 0 THEN DO:
    iPos = INDEX(cResponse, ':', iPos) + 1.
    DO WHILE SUBSTRING(cResponse, iPos, 1) = ' ' OR 
             SUBSTRING(cResponse, iPos, 1) = '"':
        iPos = iPos + 1.
    END.
    iEnd = INDEX(cResponse, '"', iPos).
    cElasticityType = SUBSTRING(cResponse, iPos, iEnd - iPos).
END.

/* Display Results */
PUT UNFORMATTED SKIP.
PUT UNFORMATTED "╔══════════════════════════════════════════════╗" SKIP.
PUT UNFORMATTED "║            ML ANALYSIS RESULTS               ║" SKIP.
PUT UNFORMATTED "╚══════════════════════════════════════════════╝" SKIP(1).

PUT UNFORMATTED "PRICING ANALYSIS:" SKIP.
PUT UNFORMATTED "  Current Price:    £" STRING(dCurrentPrice, ">>>9.99") SKIP.
PUT UNFORMATTED "  Optimal Price:    £" STRING(dOptimalPrice, ">>>9.99") SKIP.
PUT UNFORMATTED "  Price Difference: " 
    STRING(((dOptimalPrice - dCurrentPrice) / dCurrentPrice) * 100, "+>>9.9") "%" SKIP(1).

PUT UNFORMATTED "REVENUE ANALYSIS:" SKIP.
PUT UNFORMATTED "  Current Revenue:  £" STRING(dCurrentRevenue, ">>>,>>9.99") SKIP.
PUT UNFORMATTED "  Optimal Revenue:  £" STRING(dOptimalRevenue, ">>>,>>9.99") SKIP.
PUT UNFORMATTED "  Revenue Impact:   " 
    STRING(((dOptimalRevenue - dCurrentRevenue) / dCurrentRevenue) * 100, "+>>9.9") "%" SKIP(1).

PUT UNFORMATTED "ELASTICITY:" SKIP.
PUT UNFORMATTED "  Product Type: " cElasticityType SKIP(1).

/* Business Decision */
PUT UNFORMATTED "╔══════════════════════════════════════════════╗" SKIP.
PUT UNFORMATTED "║           BUSINESS RECOMMENDATION            ║" SKIP.
PUT UNFORMATTED "╚══════════════════════════════════════════════╝" SKIP(1).

IF cElasticityType = "ELASTIC" THEN DO:
    PUT UNFORMATTED "⚠ Product is PRICE SENSITIVE (Elastic)" SKIP.
    IF dOptimalPrice < dCurrentPrice THEN DO:
        PUT UNFORMATTED "→ RECOMMENDATION: REDUCE PRICE" SKIP.
        PUT UNFORMATTED "  Lower price to £" STRING(dOptimalPrice, ">>9.99") SKIP.
        PUT UNFORMATTED "  Volume increase will offset margin reduction" SKIP.
    END.
    ELSE IF dOptimalPrice > dCurrentPrice THEN DO:
        PUT UNFORMATTED "→ RECOMMENDATION: CAREFULLY INCREASE PRICE" SKIP.
        PUT UNFORMATTED "  Test price at £" STRING(dOptimalPrice, ">>9.99") SKIP.
        PUT UNFORMATTED "  Monitor volume closely - customers are price sensitive" SKIP.
    END.
    ELSE DO:
        PUT UNFORMATTED "→ RECOMMENDATION: MAINTAIN CURRENT PRICE" SKIP.
        PUT UNFORMATTED "  Current price is optimal for elastic product" SKIP.
    END.
END.
ELSE DO:  /* INELASTIC */
    PUT UNFORMATTED "✓ Product is NOT PRICE SENSITIVE (Inelastic)" SKIP.
    IF dOptimalPrice > dCurrentPrice THEN DO:
        PUT UNFORMATTED "→ RECOMMENDATION: INCREASE PRICE" SKIP.
        PUT UNFORMATTED "  Raise price to £" STRING(dOptimalPrice, ">>9.99") SKIP.
        PUT UNFORMATTED "  Customers will accept higher price" SKIP.
        PUT UNFORMATTED "  Margin improvement with minimal volume loss" SKIP.
    END.
    ELSE IF dOptimalPrice < dCurrentPrice THEN DO:
        PUT UNFORMATTED "→ RECOMMENDATION: HOLD PRICE" SKIP.
        PUT UNFORMATTED "  Keep current price despite model suggestion" SKIP.
        PUT UNFORMATTED "  Inelastic products should maximize margin" SKIP.
    END.
    ELSE DO:
        PUT UNFORMATTED "→ RECOMMENDATION: OPTIMAL PRICE ACHIEVED" SKIP.
        PUT UNFORMATTED "  Current price maximizes revenue" SKIP.
    END.
END.

PUT UNFORMATTED SKIP.
PUT UNFORMATTED "================================================" SKIP.
PUT UNFORMATTED "Session completed: " STRING(TODAY, "99/99/9999") " " STRING(TIME, "HH:MM:SS") SKIP.
PUT UNFORMATTED "Dashboard available at: http://localhost:5001/" SKIP.
PUT UNFORMATTED "================================================" SKIP.

OUTPUT CLOSE.

/* Clean up */
OS-DELETE VALUE("product.json").
OS-DELETE VALUE("request.json").
OS-DELETE VALUE("prediction.json").

/* Display summary to user */
MESSAGE "ML Price Analysis Complete!" SKIP(1)
        "Product: " cStockCode SKIP
        "Type: " cElasticityType SKIP
        "Current: £" STRING(dCurrentPrice, ">>9.99") SKIP
        "Optimal: £" STRING(dOptimalPrice, ">>9.99") SKIP
        "Revenue Impact: " STRING(((dOptimalRevenue - dCurrentRevenue) / dCurrentRevenue) * 100, "+>>9.9") "%" SKIP(1)
        "Full report: ml_session.log"
VIEW-AS ALERT-BOX INFORMATION TITLE "ML Analysis".

/* Optionally open the dashboard */
OS-COMMAND SILENT "start http://localhost:5001/".