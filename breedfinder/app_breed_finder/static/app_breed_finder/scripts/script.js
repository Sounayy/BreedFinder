let animalSelect = document.getElementById('animal-select');

animalSelect.addEventListener('change', function () {
    let form = animalSelect.closest('form');
    form.submit();
});