const express = require("express");
const cors = require("cors");
const bodyParser = require("body-parser");
const { spawn } = require("child_process");
const path = require("path");

const app = express();
const port = 5000;

app.use(cors());
app.use(bodyParser.json());

app.post("/analyze", (req, res) => {
    const { text } = req.body;
    if (!text) {
        return res.status(400).json({ error: "Text is required" });
    }

    const pythonProcess = spawn('python', [
        path.join(__dirname, 'predict_sentiment.py'),
        text
    ]);

    let result = '';
    let errorOutput = '';

    pythonProcess.stdout.on('data', (data) => {
        result += data.toString();
        console.log('Python stdout:', result); // Debug output
    });

    pythonProcess.stderr.on('data', (data) => {
        errorOutput += data.toString();
        console.error('Python stderr:', errorOutput); // Debug errors
    });

    pythonProcess.on('close', (code) => {
        if (code !== 0) {
            console.error('Python script error:', errorOutput);
            return res.status(500).json({ error: 'Sentiment analysis failed' });
        }

        try {
            console.log('Raw result from Python:', result.trim());
            const sentimentResult = JSON.parse(result.trim());
            res.json(sentimentResult);
        } catch (error) {
            console.error('Parsing error:', error.message);
            res.status(500).json({
                error: 'Unable to parse sentiment result',
                details: result.trim(), // Include raw output for debugging
            });
        }
    });
});

app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
});
