// assets/dashAgGridComponentFunctions.js

// Ensure the global namespace exists
window.dashAgGridComponentFunctions = window.dashAgGridComponentFunctions || {};
// Assign dagfuncs to it
var dagfuncs = window.dashAgGridComponentFunctions;

/**
 * Custom Cell Renderer for Action Buttons (Set Low/Set Up)
 */
dagfuncs.ActionsCellRenderer = function (props) {
    // Function to create a button element
    function createButton(text, className, actionType) {
        const button = document.createElement('button');
        button.textContent = text;
        // Use Bootstrap classes for styling
        button.classList.add('btn', 'btn-primary', 'btn-sm', 'me-1');
        if (className) {
            button.classList.add(className);
        }
        // Setting outline explicitly via style might be needed depending on CSS load order
        button.style.setProperty('--bs-btn-border-color', 'var(--bs-primary)');
        button.style.setProperty('--bs-btn-hover-bg', 'var(--bs-primary)');
        button.style.setProperty('--bs-btn-hover-color', 'var(--bs-white)');
        button.style.setProperty('--bs-btn-color', 'var(--bs-primary)');
        button.style.backgroundColor = 'transparent'; // For outline effect

        // Add event listener
        button.addEventListener('click', function() {
            console.log('Button clicked:', actionType, 'for sample:', props.data.sample_id);
            // Trigger a Dash callback by updating the hidden store
            // Pass necessary data: action type and sample_id
            // The Dash callback will listen to changes in 'grid-action-trigger'

            // Find the setProps function - requires Dash client-side context
            // This assumes default Dash clientside context setup
            const setProps = Dash.setProps ? Dash.setProps : (id, props) => {
                 const component = Dash.registry.find(c => c.id === id);
                 if (component) {
                      component.setProps(props);
                 } else {
                      console.error(`Component with id ${id} not found`);
                 }
            };

            if (setProps) {
                setProps('grid-action-trigger', {
                    data: {
                        action: actionType,
                        sample_id: props.data.sample_id,
                        timestamp: Date.now() // Add timestamp to ensure trigger
                    }
                });
            } else {
                console.error("Dash clientside setProps not available for triggering callback.");
            }
        });
        return button;
    }

    // Create the div container for the buttons
    const eDiv = document.createElement('div');
    eDiv.appendChild(createButton('Set Low', 'btn-outline-primary', 'lower'));
    eDiv.appendChild(createButton('Set Up', 'btn-outline-primary', 'upper'));

    return eDiv;
}

/**
 * Custom Cell Renderer for Checkboxes (Reviewed/Exclude) - Returning Simple Character
 */
dagfuncs.CheckboxRenderer = function (props) {
    const isChecked = props.value;
    // Return a simple unicode checkmark or empty string
    return isChecked ? '✓' : '';
}
