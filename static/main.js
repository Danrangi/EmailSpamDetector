document.addEventListener('DOMContentLoaded', () => {
    // 1. Fetch backend data from the hidden HTML div
    const appData = document.getElementById('app-data');
    const serverPrediction = appData.dataset.prediction;
    const serverEmailText = appData.dataset.emailText;
    const serverConfidence = parseFloat(appData.dataset.confidence || 0);

    // 2. Setup variables
    const avatarColors = ['#2563eb','#d97706','#7c3aed','#059669','#dc2626','#0891b2','#c026d3','#ea580c'];
    let inboxMessages = [];
    let spamMessages = [];
    let activeTab = 'compose';
    let selectedId = null;
    let idCounter = 1;

    // 3. Helper Functions
    function getInitials(text) {
        return text.substring(0, 2).toUpperCase();
    }

    function getColor(id) {
        return avatarColors[id % avatarColors.length];
    }

    function now() {
        const d = new Date();
        let h = d.getHours(), m = d.getMinutes();
        const ampm = h >= 12 ? 'PM' : 'AM';
        h = h % 12 || 12;
        return h + ':' + (m < 10 ? '0' : '') + m + ' ' + ampm;
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // 4. Initial processing from Server Prediction
    if (serverPrediction && serverPrediction !== 'unknown') {
        const firstLine = serverEmailText.split('\n')[0].substring(0, 60) || 'No subject';
        const preview = serverEmailText.substring(0, 120);
        const msg = {
            id: idCounter++,
            sender: serverPrediction === 'spam' ? 'Suspicious Sender' : 'Unknown Sender',
            subject: firstLine,
            preview: preview,
            body: serverEmailText,
            time: now(),
            read: false,
            confidence: serverConfidence,
            prediction: serverPrediction
        };
        if (serverPrediction === 'spam') {
            spamMessages.unshift(msg);
        } else {
            inboxMessages.unshift(msg);
        }
    }

    // 5. Local Storage Management
    function loadMessages() {
        try {
            const saved = localStorage.getItem('mailguard_inbox');
            if (saved) inboxMessages = [...JSON.parse(saved), ...inboxMessages];
            const savedSpam = localStorage.getItem('mailguard_spam');
            if (savedSpam) spamMessages = [...JSON.parse(savedSpam), ...spamMessages];
            
            // Deduplicate by body
            const seenInbox = new Set();
            inboxMessages = inboxMessages.filter(m => {
                if (seenInbox.has(m.body)) return false;
                seenInbox.add(m.body); return true;
            });
            
            const seenSpam = new Set();
            spamMessages = spamMessages.filter(m => {
                if (seenSpam.has(m.body)) return false;
                seenSpam.add(m.body); return true;
            });
            
            let c = 1;
            inboxMessages.forEach(m => m.id = c++);
            spamMessages.forEach(m => m.id = c++);
            idCounter = c;
        } catch(e) {}
    }

    function saveMessages() {
        try {
            localStorage.setItem('mailguard_inbox', JSON.stringify(inboxMessages));
            localStorage.setItem('mailguard_spam', JSON.stringify(spamMessages));
        } catch(e) {}
    }

    function updateCounts() {
        document.getElementById('inbox-count').textContent = inboxMessages.length;
        document.getElementById('spam-count').textContent = spamMessages.length;
    }

    // 6. UI Rendering
    window.switchTab = function(tab) {
        activeTab = tab;
        selectedId = null;

        document.getElementById('nav-inbox').classList.toggle('active', tab === 'inbox');
        document.getElementById('nav-spam').classList.toggle('active', tab === 'spam');
        document.getElementById('nav-compose').classList.toggle('active', tab === 'compose');

        const composeView = document.getElementById('compose-view');
        const detailView = document.getElementById('email-detail-view');
        const panelTitle = document.getElementById('panel-title');
        const panelSub = document.getElementById('panel-sub');

        if (tab === 'compose') {
            composeView.style.display = 'block';
            detailView.style.display = 'none';
            panelTitle.textContent = 'History';
            const all = [...inboxMessages, ...spamMessages];
            panelSub.textContent = all.length + ' total message' + (all.length !== 1 ? 's' : '');
            renderList(all.sort((a,b) => b.id - a.id));
        } else if (tab === 'inbox') {
            composeView.style.display = 'none';
            detailView.style.display = 'none';
            panelTitle.textContent = 'Inbox';
            panelSub.textContent = inboxMessages.length + ' message' + (inboxMessages.length !== 1 ? 's' : '');
            renderList(inboxMessages);
            showPlaceholder();
        } else {
            composeView.style.display = 'none';
            detailView.style.display = 'none';
            panelTitle.textContent = 'Spam';
            panelSub.textContent = spamMessages.length + ' message' + (spamMessages.length !== 1 ? 's' : '') + (spamMessages.length > 0 ? ' — be careful' : '');
            renderList(spamMessages);
            showPlaceholder();
        }
    };

    function showPlaceholder() {
        const dv = document.getElementById('email-detail-view');
        dv.style.display = 'flex';
        dv.innerHTML = `
            <div class="detail-placeholder">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" stroke-linecap="round" stroke-linejoin="round">
                    <rect x="2" y="4" width="20" height="16" rx="2"/>
                    <path d="m22 7-8.97 5.7a1.94 1.94 0 0 1-2.06 0L2 7"/>
                </svg>
                <p>Select a message to read</p>
                <span>${activeTab === 'inbox' ? inboxMessages.length + ' conversations in your inbox' : spamMessages.length + ' messages flagged as spam'}</span>
            </div>
        `;
    }

    function renderList(messages) {
        const listEl = document.getElementById('message-list');
        const clearBtn = document.getElementById('clear-list-btn');
        
        if (messages.length > 0) {
            clearBtn.style.display = 'block';
        } else {
            clearBtn.style.display = 'none';
        }

        if (messages.length === 0) {
            listEl.innerHTML = `
                <div class="empty-state">
                    <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                        <rect x="2" y="4" width="20" height="16" rx="2"/>
                        <path d="m22 7-8.97 5.7a1.94 1.94 0 0 1-2.06 0L2 7"/>
                    </svg>
                    <p>Nothing here</p>
                    <span>${activeTab === 'inbox' ? "You're all clear" : activeTab === 'spam' ? 'No spam detected' : 'Classify emails to see them here'}</span>
                </div>
            `;
            return;
        }

        listEl.innerHTML = messages.map(msg => {
            const isSpam = msg.prediction === 'spam';
            const color = isSpam ? '#dc2626' : getColor(msg.id);
            const initials = getInitials(msg.sender);
            return `
                <div class="message-item ${selectedId === msg.id ? 'selected' : ''}" onclick="selectMessage(${msg.id})">
                    ${!msg.read ? '<div class="msg-unread-dot"></div>' : ''}
                    <div class="msg-avatar" style="background:${color}">${initials}</div>
                    <div class="msg-content">
                        <div class="msg-top">
                            <span class="msg-sender ${msg.read ? 'read' : ''}">${escapeHtml(msg.sender)}</span>
                            <span class="msg-time">${msg.time}</span>
                        </div>
                        <div class="msg-subject">${escapeHtml(msg.subject)}</div>
                        <div class="msg-preview">${escapeHtml(msg.preview)}</div>
                    </div>
                    <button onclick="event.stopPropagation(); deleteMessage(${msg.id})" style="background:transparent; border:none; color:#64748b; cursor:pointer; padding:4px; display:flex; align-items:center;">
                        <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <polyline points="3 6 5 6 21 6"></polyline>
                            <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                        </svg>
                    </button>
                </div>
            `;
        }).join('');
    }

    // 7. Actions
    window.selectMessage = function(id) {
        selectedId = id;
        const allMsgs = [...inboxMessages, ...spamMessages];
        const msg = allMsgs.find(m => m.id === id);
        if (!msg) return;

        msg.read = true;
        saveMessages();

        const isSpam = msg.prediction === 'spam';
        const color = isSpam ? '#dc2626' : getColor(msg.id);

        document.getElementById('compose-view').style.display = 'none';
        const dv = document.getElementById('email-detail-view');
        dv.style.display = 'flex';

        const warningHtml = isSpam ? `
            <div class="spam-warning">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/>
                    <path d="M12 9v4"/><path d="M12 17h.01"/>
                </svg>
                <p>This message was identified as spam. Be cautious with any links or attachments.</p>
            </div>
        ` : '';

        const moveBtn = isSpam
            ? `<button class="btn-action move-inbox" onclick="moveToInbox(${id})">Not Spam</button>`
            : `<button class="btn-action move-spam" onclick="moveToSpam(${id})">Report Spam</button>`;

        dv.innerHTML = `
            <div class="detail-email-header-wrap">
                <div style="flex:1; min-width:0;">
                    <div class="detail-email-subject">${escapeHtml(msg.subject)}</div>
                    <div class="detail-email-meta">
                        <div class="msg-avatar" style="background:${color};width:28px;height:28px;font-size:10px;">${getInitials(msg.sender)}</div>
                        <span class="meta-sender">${escapeHtml(msg.sender)}</span>
                        <span class="meta-time">· ${msg.time}</span>
                        <span class="meta-badge" style="background:${isSpam ? 'rgba(239, 68, 68, 0.1)' : 'rgba(34, 197, 94, 0.1)'};color:${isSpam ? '#f87171' : '#4ade80'};">${isSpam ? 'Spam' : 'Safe'} · ${msg.confidence}%</span>
                    </div>
                </div>
                <div class="detail-actions">
                    ${moveBtn}
                    <button class="btn-action delete" onclick="deleteMessage(${id})">Delete</button>
                </div>
            </div>
            <div class="detail-email">
                ${warningHtml}
                <div class="detail-email-body">${escapeHtml(msg.body)}</div>
            </div>
        `;

        if (activeTab === 'inbox') renderList(inboxMessages);
        else if (activeTab === 'spam') renderList(spamMessages);
        else renderList([...inboxMessages, ...spamMessages].sort((a,b) => b.id - a.id));
    };

    window.moveToSpam = function(id) {
        const idx = inboxMessages.findIndex(m => m.id === id);
        if (idx > -1) {
            const msg = inboxMessages.splice(idx, 1)[0];
            msg.prediction = 'spam';
            spamMessages.unshift(msg);
            finalizeAction();
        }
    };

    window.moveToInbox = function(id) {
        const idx = spamMessages.findIndex(m => m.id === id);
        if (idx > -1) {
            const msg = spamMessages.splice(idx, 1)[0];
            msg.prediction = 'ham';
            inboxMessages.unshift(msg);
            finalizeAction();
        }
    };

    window.deleteMessage = function(id) {
        inboxMessages = inboxMessages.filter(m => m.id !== id);
        spamMessages = spamMessages.filter(m => m.id !== id);
        finalizeAction();
    };

    window.clearCurrentList = function() {
        if (!confirm("Are you sure you want to delete all messages in this view?")) return;
        
        if (activeTab === 'inbox') {
            inboxMessages = [];
        } else if (activeTab === 'spam') {
            spamMessages = [];
        } else {
            inboxMessages = [];
            spamMessages = [];
        }
        
        finalizeAction();
    };

    function finalizeAction() {
        saveMessages();
        updateCounts();
        selectedId = null;
        switchTab(activeTab);
    }

    // 8. Boot Sequence
    loadMessages();
    updateCounts();

    if (serverPrediction && serverPrediction !== 'unknown') {
        saveMessages();
        updateCounts();
        switchTab(serverPrediction === 'spam' ? 'spam' : 'inbox');
    } else {
        switchTab('compose');
    }
});