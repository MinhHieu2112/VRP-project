document.addEventListener('DOMContentLoaded', () => {
    const runForm = document.getElementById('runForm');
    const runBtn = document.getElementById('runBtn');
    const statusContainer = document.getElementById('statusContainer');
    const statusText = document.getElementById('statusText');
    
    const metricsGrid = document.getElementById('metricsGrid');
    const mapPlaceholder = document.getElementById('mapPlaceholder');
    const mapFrame = document.getElementById('mapFrame');
    const routesContainer = document.getElementById('routesContainer');
    const routesList = document.getElementById('routesList');
    
    const valDistance = document.getElementById('valDistance');
    const valTime = document.getElementById('valTime');
    const valVehicles = document.getElementById('valVehicles');
    const valCustomers = document.getElementById('valCustomers');

    // --- SETUP CUSTOM SELECT ---
    function setupCustomSelect(wrapperId, displayId, optionsId, textId, inputId) {
        const wrapper = document.getElementById(wrapperId);
        const display = document.getElementById(displayId);
        const textEl = document.getElementById(textId);
        const inputEl = document.getElementById(inputId);
        const optionsContainer = document.getElementById(optionsId);
        
        display.addEventListener('click', (e) => {
            document.querySelectorAll('.custom-select-wrapper').forEach(w => {
                if (w !== wrapper) w.classList.remove('open');
            });
            wrapper.classList.toggle('open');
            e.stopPropagation();
        });

        optionsContainer.addEventListener('click', (e) => {
            const option = e.target.closest('.custom-option');
            if (option) {
                optionsContainer.querySelectorAll('.custom-option').forEach(opt => opt.classList.remove('selected'));
                option.classList.add('selected');
                inputEl.value = option.dataset.value;
                textEl.textContent = option.textContent.trim();
                wrapper.classList.remove('open');
            }
        });
    }

    setupCustomSelect('algoDropdownWrapper', 'algoDisplay', 'algoOptions', 'algoSelectedText', 'algorithm');
    setupCustomSelect('pointsDropdownWrapper', 'pointsDisplay', 'pointsOptions', 'pointsSelectedText', 'numPoints');

    document.addEventListener('click', () => {
        document.querySelectorAll('.custom-select-wrapper').forEach(w => w.classList.remove('open'));
    });

    // --- FETCH ALGORITHMS ---
    fetch('/api/algorithms')
        .then(res => res.json())
        .then(data => {
            const algoOptions = document.getElementById('algoOptions');
            const algoInput = document.getElementById('algorithm');
            const algoText = document.getElementById('algoSelectedText');
            
            algoOptions.innerHTML = '';
            
            if (data.length > 0) {
                algoInput.value = data[0].id;
                algoText.textContent = data[0].name;
            }
            
            data.forEach((algo, index) => {
                const opt = document.createElement('div');
                opt.className = 'custom-option' + (index === 0 ? ' selected' : '');
                opt.dataset.value = algo.id;
                opt.textContent = algo.name;
                algoOptions.appendChild(opt);
            });
        })
        .catch(err => {
            document.getElementById('algoSelectedText').textContent = 'Lỗi tải danh sách';
        });

    // --- PAGINATION LOGIC ---
    let allRoutes = [];
    let currentPage = 1;
    const routesPerPage = 6;
    
    document.getElementById('prevPageBtn').addEventListener('click', () => {
        if (currentPage > 1) {
            currentPage--;
            renderRoutesPage();
        }
    });

    document.getElementById('nextPageBtn').addEventListener('click', () => {
        const totalPages = Math.ceil(allRoutes.length / routesPerPage) || 1;
        if (currentPage < totalPages) {
            currentPage++;
            renderRoutesPage();
        }
    });

    function renderRoutesPage() {
        routesList.innerHTML = '';
        const start = (currentPage - 1) * routesPerPage;
        const end = start + routesPerPage;
        const pageRoutes = allRoutes.slice(start, end);
        
        pageRoutes.forEach(routeData => {
            const routeEl = document.createElement('div');
            routeEl.className = 'route-item';
            
            const header = document.createElement('div');
            header.className = 'route-header';
            header.innerHTML = `<span class="route-id">Xe #${routeData.id.padStart(3, '0')}</span> <span class="route-stops">${routeData.path.length} điểm</span>`;
            
            const pathInfo = document.createElement('div');
            pathInfo.className = 'route-path';
            pathInfo.textContent = routeData.path.join(' → ');
            
            routeEl.appendChild(header);
            routeEl.appendChild(pathInfo);
            routesList.appendChild(routeEl);
        });
        
        const totalPages = Math.ceil(allRoutes.length / routesPerPage) || 1;
        document.getElementById('pageIndicator').textContent = `Trang ${currentPage} / ${totalPages}`;
        document.getElementById('prevPageBtn').disabled = currentPage === 1;
        document.getElementById('nextPageBtn').disabled = currentPage === totalPages;
    }

    // --- FORM SUBMIT ---
    runForm.addEventListener('submit', (e) => {
        e.preventDefault();
        
        const algoId = document.getElementById('algorithm').value;
        const numPoints = document.getElementById('numPoints').value;
        
        if (!algoId) return;

        runBtn.disabled = true;
        statusContainer.style.display = 'flex';
        metricsGrid.style.display = 'none';
        mapFrame.style.display = 'none';
        mapPlaceholder.style.display = 'flex';
        routesContainer.style.display = 'none';
        
        // Ẩn terminal log cũ
        const logContainer = document.getElementById('logContainer');
        if(logContainer) logContainer.style.display = 'none';
        
        const messages = [
            "Khởi tạo thuật toán...",
            "Tìm kiếm vùng lân cận...",
            "Tối ưu hoá lộ trình...",
            "Đang hội tụ nghiệm..."
        ];
        let msgIndex = 0;
        statusText.textContent = messages[0];
        
        const msgInterval = setInterval(() => {
            msgIndex = (msgIndex + 1) % messages.length;
            statusText.textContent = messages[msgIndex];
        }, 2000);

        fetch('/api/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                algorithm_id: algoId,
                num_points: parseInt(numPoints)
            })
        })
        .then(res => res.json().then(data => ({ status: res.status, body: data })))
        .then(({ status, body }) => {
            clearInterval(msgInterval);
            if (status !== 200) throw new Error(body.error || 'Lỗi không xác định');
            displayResults(body);
        })
        .catch(err => {
            clearInterval(msgInterval);
            alert(`Lỗi: ${err.message}`);
        })
        .finally(() => {
            runBtn.disabled = false;
            statusContainer.style.display = 'none';
        });
    });

    function displayResults(data) {
        valDistance.textContent = data.total_distance_km.toFixed(2);
        valTime.textContent = data.execution_time.toFixed(2);
        valVehicles.textContent = data.num_vehicles;
        valCustomers.textContent = data.num_customers || 0;
        
        metricsGrid.style.display = 'grid';
        
        mapPlaceholder.style.display = 'none';
        mapFrame.src = `${data.map_url}?t=${new Date().getTime()}`;
        mapFrame.style.display = 'block';
        
        // Prepare routes for pagination
        allRoutes = [];
        const routes = data.routes;
        for (const [vId, path] of Object.entries(routes)) {
            const cleanPath = path.filter(n => n !== 0);
            if (cleanPath.length === 0) continue;
            allRoutes.push({ id: vId, path: cleanPath });
        }
        
        currentPage = 1;
        renderRoutesPage();
        routesContainer.style.display = 'block';

        // Hiển thị terminal log
        const logContainer = document.getElementById('logContainer');
        const logContent = document.getElementById('logContent');
        if (data.log) {
            logContent.textContent = data.log;
            logContainer.style.display = 'flex';
            // Scroll to bottom
            logContent.scrollTop = logContent.scrollHeight;
        } else {
            logContainer.style.display = 'none';
        }
    }
    
    // Tính năng phóng to/thu nhỏ Terminal Log
    const logExpandBtn = document.getElementById('logExpandBtn');
    if(logExpandBtn) {
        logExpandBtn.addEventListener('click', () => {
            const logContainer = document.getElementById('logContainer');
            logContainer.classList.toggle('fullscreen');
            
            // Đổi icon
            if (logContainer.classList.contains('fullscreen')) {
                logExpandBtn.innerHTML = '<i class="ph-bold ph-arrows-in"></i>';
            } else {
                logExpandBtn.innerHTML = '<i class="ph-bold ph-arrows-out"></i>';
            }
        });
    }
});
