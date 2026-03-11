function datetime_str = getDateTime()
    % Cette fonction retourne une chaîne de caractères indiquant la date et l'heure actuelles
    datetime_str = string(datetime('now', 'Format', 'yyyy-MM-dd_HH-mm-ss'));
end