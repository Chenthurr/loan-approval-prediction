(function() {
    // Tab switching
    const tabs = document.querySelectorAll('.tab-btn');
    const panels = document.querySelectorAll('.panel');
    tabs.forEach(t => {
        t.addEventListener('click', () => {
            tabs.forEach(x => x.classList.remove('active'));
            panels.forEach(x => x.classList.remove('active'));
            t.classList.add('active');
            document.getElementById(t.dataset.tab).classList.add('active');
        });
    });

    // Range slider updates
    const cs = document.getElementById('creditScore');
    const csVal = document.getElementById('csVal');
    const emp = document.getElementById('empLength');
    const empVal = document.getElementById('empVal');
    const dti = document.getElementById('dti');
    const dtiVal = document.getElementById('dtiVal');

    cs.addEventListener('input', () => csVal.textContent = cs.value);
    emp.addEventListener('input', () => empVal.textContent = emp.value);
    dti.addEventListener('input', () => dtiVal.textContent = dti.value);

    // Feature importance data
    const features = [
        { name: 'Credit score', value: 31, color: '#58a6ff' },
        { name: 'Debt-to-income', value: 22, color: '#f85149' },
        { name: 'Annual income', value: 18, color: '#3fb950' },
        { name: 'Loan amount', value: 14, color: '#a371f7' },
        { name: 'Employment', value: 9, color: '#8b949e' },
        { name: 'Home ownership', value: 6, color: '#6e7681' }
    ];

    const fl = document.getElementById('featureList');
    features.forEach(f => {
        const row = document.createElement('div');
        row.className = 'feature-row';
        row.innerHTML = `
            <div class="feature-name">${f.name}</div>
            <div class="feature-track"><div class="feature-fill" style="width:0%;background:${f.color}"></div></div>
            <div class="feature-value">${f.value}%</div>
        `;
        fl.appendChild(row);
        setTimeout(() => row.querySelector('.feature-fill').style.width = f.value + '%', 100);
    });

    // Purpose distribution chart
    const purposes = [
        { label: 'Debt consol.', value: 42, color: '#58a6ff' },
        { label: 'Home improv.', value: 18, color: '#f85149' },
        { label: 'Medical', value: 14, color: '#3fb950' },
        { label: 'Education', value: 12, color: '#a371f7' },
        { label: 'Business', value: 9, color: '#8b949e' },
        { label: 'Other', value: 5, color: '#6e7681' }
    ];
    const pc = document.getElementById('purposeChart');
    purposes.forEach(p => {
        const item = document.createElement('div');
        item.className = 'bar-item';
        item.innerHTML = `
            <div class="bar-rect" style="height:0%;background:${p.color}"></div>
            <div class="bar-label">${p.label}</div>
            <div class="bar-label">${p.value}%</div>
        `;
        pc.appendChild(item);
        setTimeout(() => item.querySelector('.bar-rect').style.height = (p.value * 2.6) + 'px', 200);
    });

    // Prediction logic
    document.getElementById('predictionForm').addEventListener('submit', function(e) {
        e.preventDefault();

        const income = parseFloat(document.getElementById('income').value) || 0;
        const loan = parseFloat(document.getElementById('loanAmount').value) || 0;
        const credit = parseInt(cs.value);
        const empLen = parseInt(emp.value);
        const dtiVal = parseInt(dti.value);
        const home = document.getElementById('homeOwnership').value;
        const purpose = document.getElementById('loanPurpose').value;
        const defaults = parseInt(document.getElementById('prevDefaults').value);

        // Risk scoring model (mirrors Python logic)
        let score = 0.5;
        score += (credit - 600) / 500 * 0.25;
        score += (income - 50000) / 100000 * 0.15;
        score -= (loan - 15000) / 50000 * 0.15;
        score -= (dtiVal - 20) / 60 * 0.2;
        score += (empLen / 30) * 0.08;
        if (home === 'own') score += 0.05;
        if (home === 'mortgage') score += 0.02;
        score -= defaults * 0.12;
        if (purpose === 'business') score -= 0.03;
        if (purpose === 'medical') score += 0.02;

        score = Math.max(0.02, Math.min(0.98, score));
        const prob = Math.round(score * 100);
        const approved = prob >= 55;

        const resultBox = document.getElementById('resultBox');
        const resultStatus = document.getElementById('resultStatus');
        const gaugeFill = document.getElementById('gaugeFill');
        const gaugeText = document.querySelector('.gauge-text');
        const riskBadge = document.getElementById('riskBadge');
        const confBadge = document.getElementById('confBadge');

        resultBox.classList.add('show');
        resultStatus.textContent = approved ? 'Approved' : 'Rejected';
        resultStatus.className = 'result-status ' + (approved ? 'approved' : 'rejected');

        const offset = 251 - (251 * prob / 100);
        gaugeFill.style.strokeDashoffset = offset;
        gaugeFill.style.stroke = approved ? '#3fb950' : '#f85149';
        gaugeText.textContent = prob + '%';
        gaugeText.style.fill = approved ? '#3fb950' : '#f85149';

        let risk = 'low', riskText = 'Low Risk';
        if (prob < 40) { risk = 'high'; riskText = 'High Risk'; }
        else if (prob < 70) { risk = 'medium'; riskText = 'Medium Risk'; }
        riskBadge.className = 'risk-badge ' + risk;
        riskBadge.textContent = riskText;

        const conf = approved ? prob : (100 - prob);
        confBadge.textContent = 'Confidence: ' + conf + '%';
    });
})();
